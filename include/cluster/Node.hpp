#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "net/Protocol.hpp"
#include "net/Socket.hpp"
#include "runtime/Runtime.hpp"

struct NodeConfig {
    std::string coordinator_host = "127.0.0.1";
    uint16_t    coordinator_port = 9000;
    uint16_t    listen_port      = 0;      // 0 = let the kernel pick
    std::string label            = "cpu";
    int         num_workers      = 4;
    bool        enable_stealing  = true;   // off = static assignment, for comparison
    int         heartbeat_ms     = 300;
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
    uint64_t steal_requests_sent() const { return steal_requests_.load(std::memory_order_relaxed); }

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
    };

    void accept_loop();
    void serve_peer(TcpSocket conn);
    void coordinator_loop();
    void steal_loop();
    void completion_loop();

    bool register_with_coordinator();
    void update_peers(const proto::NodeListMsg& msg);

    // Runs one remote steal attempt against a randomly chosen peer.
    bool try_remote_steal();

    // Turn a task received from another node into a local closure task that reports
    // completion home when it finishes.
    void accept_stolen_task(const Task& remote);

    void enqueue_completion(uint32_t origin_node, uint64_t task_id);

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

    std::atomic<uint64_t> stolen_in_{0};
    std::atomic<uint64_t> stolen_out_{0};
    std::atomic<uint64_t> steal_requests_{0};

    std::mt19937 rng_{std::random_device{}()};
};
