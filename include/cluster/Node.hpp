#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "net/Protocol.hpp"
#include "net/Socket.hpp"
#include "runtime/Runtime.hpp"

// How a node picks whom to steal from. Exposed as a knob so the policies can be
// measured against each other rather than asserted to be better.
enum class StealPolicy {
    None,       // never steal — static assignment, the baseline
    Random,     // uniformly random victim: the classic work-stealing choice
    LoadAware,  // prefer the peer reporting the most queued work
    Adaptive,   // load + capability affinity + backoff tuned by observed success
};

const char* steal_policy_name(StealPolicy p);
bool        parse_steal_policy(const std::string& s, StealPolicy& out);

struct NodeConfig {
    std::string coordinator_host = "127.0.0.1";
    uint16_t    coordinator_port = 9000;
    uint16_t    listen_port      = 0;      // 0 = let the kernel pick
    // Address peers should use to reach this node. Must be an address other machines
    // can actually route to when running across a real network — "127.0.0.1" only
    // works when the whole cluster is on one host.
    std::string advertise_host   = "127.0.0.1";
    std::string label            = "cpu";
    int         num_workers      = 4;
    StealPolicy policy           = StealPolicy::Adaptive;
    int         heartbeat_ms     = 300;
    // A task handed to a peer that has not reported back within this window is assumed
    // lost and re-run locally. Must comfortably exceed the slowest expected task, or
    // healthy work gets duplicated.
    int         task_timeout_ms  = 4000;
    // Fault injection for testing: when set, this node runs stolen tasks but never
    // reports them home — exactly what the origin observes when a node dies after
    // taking work. Never enable in production.
    bool        drop_completions = false;
};

// A single machine in the cluster: a local Runtime plus the networking that lets other
// nodes steal from it.
//
// Threading: worker threads never touch the network. A dedicated steal thread issues
// remote steal requests and a sender thread reports completions, because a blocking
// socket call on a worker thread would stall that worker's entire share of the machine.
class Node {
public:
    explicit Node(NodeConfig cfg);
    ~Node();

    bool start();
    void stop();

    Runtime& runtime() { return *runtime_; }

    uint32_t node_id() const { return node_id_.load(std::memory_order_relaxed); }
    uint16_t port() const    { return listen_.local_port(); }

    uint64_t tasks_stolen_in() const  { return stolen_in_.load(std::memory_order_relaxed); }
    uint64_t tasks_stolen_out() const { return stolen_out_.load(std::memory_order_relaxed); }
    // Tasks re-run locally after the peer they were given to went silent.
    uint64_t tasks_reassigned() const { return tasks_reassigned_.load(std::memory_order_relaxed); }
    std::size_t outstanding_count() const;
    uint64_t steal_requests_sent() const { return steal_requests_.load(std::memory_order_relaxed); }
    uint64_t steal_misses() const { return steal_misses_.load(std::memory_order_relaxed); }

    // Fraction of steal requests that actually returned work. The signal the Adaptive
    // policy uses, and the one worth reporting when comparing policies.
    double steal_success_rate() const {
        uint64_t sent = steal_requests_sent();
        return sent ? double(sent - steal_misses()) / double(sent) : 0.0;
    }

    // Declares which task type this node runs fastest. Set from the node's label by
    // the application (e.g. a "gpu" node prefers inference work).
    void set_preferred_type(TaskType t) { preferred_type_ = t; }

    std::size_t peer_count() const;

private:
    // One outbound connection per peer. Both the steal thread and the completion sender
    // write to it, so it carries its own mutex; a request/response pair is held under
    // the lock so replies can't be interleaved between the two users.
    struct PeerLink {
        proto::NodeInfo info;
        TcpSocket       sock;
        std::mutex      mutex;
        bool            connected = false;
        // Consecutive empty/failed steals. Used by the Adaptive policy to stop
        // hammering a peer that has nothing to give.
        std::atomic<uint32_t> misses{0};
    };

    void accept_loop();
    void serve_peer(TcpSocket conn);
    void coordinator_loop();
    void steal_loop();
    void completion_loop();

    bool register_with_coordinator();
    void update_peers(const proto::NodeListMsg& msg);

    // Runs one remote steal attempt. Which peer is chosen depends on cfg_.policy.
    bool try_remote_steal();

    // Applies the configured policy to pick whom to ask. Returns nullptr if there is
    // no sensible victim.
    std::shared_ptr<PeerLink> select_victim();

    // Turn a task received from another node into a local closure task that reports
    // completion home when it finishes.
    void accept_stolen_task(const Task& remote);

    void enqueue_completion(uint32_t origin_node, uint64_t task_id);

    // ── fault tolerance (Phase 10) ──────────────────────────────────────────
    // Tasks this node handed to a thief and is still waiting to hear back about.
    // Without this a dead thief means the origin's wait_all() never returns: the task
    // was counted as submitted, and the only thing that would ever complete it is a
    // TaskResult that is never coming.
    struct OutstandingTask {
        Task     task;        // kept so it can be re-run if the thief dies
        uint32_t thief_node = 0;
        int64_t  sent_ms    = 0;
    };

    void track_outstanding(const Task& t, uint32_t thief_node);
    void resolve_outstanding(uint64_t task_id);
    // Re-queues tasks whose thief has gone silent. Runs on the reaper thread.
    void reap_outstanding();
    void reaper_loop();

    int64_t now_ms() const;

    std::shared_ptr<PeerLink> peer_for(uint32_t node_id);

    NodeConfig               cfg_;
    std::unique_ptr<Runtime> runtime_;

    TcpSocket             listen_;
    TcpSocket             coord_sock_;
    std::atomic<uint32_t> node_id_{0};
    std::atomic<bool>     running_{false};

    std::thread              accept_thread_;
    std::thread              coord_thread_;
    std::thread              steal_thread_;
    std::thread              completion_thread_;
    std::vector<std::thread> peer_threads_;
    std::mutex               peer_threads_mutex_;

    mutable std::mutex                     peers_mutex_;
    std::vector<std::shared_ptr<PeerLink>> peers_;

    // Completion notices waiting to be sent home. The origin node's wait_all() cannot
    // return until these arrive, so they are pushed out on a condition variable rather
    // than polled — a polling delay here shows up directly as extra job latency.
    std::mutex                                completions_mutex_;
    std::condition_variable                   completions_cv_;
    std::deque<std::pair<uint32_t, uint64_t>> completions_;

    std::thread                                    reaper_thread_;
    mutable std::mutex                             outstanding_mutex_;
    std::unordered_map<uint64_t, OutstandingTask>  outstanding_;
    std::chrono::steady_clock::time_point          origin_time_ = std::chrono::steady_clock::now();

    std::atomic<uint64_t> stolen_in_{0};
    std::atomic<uint64_t> stolen_out_{0};
    std::atomic<uint64_t> tasks_reassigned_{0};
    std::atomic<uint64_t> steal_requests_{0};
    // Requests that came back empty. The ratio against steal_requests_ is the steal
    // success rate the Adaptive policy tunes its backoff against.
    std::atomic<uint64_t> steal_misses_{0};
    // Consecutive failed steals, reset on any success. Drives the adaptive backoff.
    std::atomic<uint32_t> consecutive_failures_{0};

    // What this node is good at, derived from its label. Sent with every steal request
    // so victims can hand over work this node will run fastest.
    TaskType preferred_type_ = TaskType::Count;

    std::mt19937 rng_{std::random_device{}()};
};
