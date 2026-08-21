#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/InjectionQueue.hpp"
#include "runtime/Task.hpp"
#include "runtime/WorkDeque.hpp"
#include "runtime/Worker.hpp"
#include "metrics/Metrics.hpp"
#include "Futures/Future.hpp"

class Runtime {
public:
    // enable_tracing turns on per-task latency sampling. Off gives a pure throughput
    // measurement with no clock reads on the task path.
    explicit Runtime(int num_workers, bool enable_tracing = true);
    ~Runtime();

    // wrap a callable into a Task and push it onto the global queue
    void submit(std::function<void()> fn);

    // Submit a task in portable form (type tag + payload) so it can be executed here
    // or stolen by another node. Requires a handler registered for `type`.
    void submit_portable(TaskType type, std::vector<uint8_t> payload, uint32_t cost_hint = 1);

    // Submit an already-formed Task to this node's local queues. Used to re-run a task
    // recovered from a failed peer: the task keeps its payload (so a pipeline observer
    // can still advance the chain) but carries an `fn`, which makes it non-portable and
    // therefore not stealable again by the peer that just dropped it.
    void submit_task(Task t);

    // Hand a portable task to a remote thief. Returns false if none is available.
    // The task stays counted as submitted here; the thief reports its completion back.
    //
    // `preferred` biases which kind of work is given away: a thief that is fast at one
    // task type gets that type first, falling back to anything else. Pass
    // TaskType::Count for no preference.
    bool take_portable(Task& out, TaskType preferred = TaskType::Count);

    // How many portable tasks are currently available to steal.
    std::size_t portable_available() const;

    // Called when a node reports that a task it stole from us has finished. `t` is the
    // task as it was handed out, so a pipeline can advance to its next stage even
    // though this node never ran it.
    void on_remote_task_complete(const Task& t);

    // Invoked after every task finishes, with the task itself, before the task is
    // counted as complete. A multi-stage job uses this to submit its next stage.
    // Set once before the runtime starts taking work.
    void set_task_observer(std::function<void(const Task&)> fn) {
        task_observer_ = std::move(fn);
    }

    // Identifies this Runtime's node so stolen tasks know where to report completion.
    void     set_node_id(uint32_t id) { node_id_ = id; }
    uint32_t node_id() const          { return node_id_; }

    uint64_t pending() const {
        return submitted_.load(std::memory_order_relaxed) -
               completed_.load(std::memory_order_relaxed);
    }

    // try to pop and execute one task from any queue — returns true if a task was executed
    bool try_execute_one();

    template<typename T>
    std::shared_ptr<Future<T>> spawn(std::function<T()> fn) {
        auto f = std::make_shared<Future<T>>(
            [this]() { return try_execute_one(); }
        );
        spawned_.fetch_add(1, std::memory_order_relaxed);
        submit([f, fn]() {
            f->set(fn());
        });
        return f;
    }

    // block until every submitted task has completed
    void wait_all();

    // signal all workers to stop and join their threads
    void shutdown();

    // called by each worker after it finishes executing a task
    void on_task_complete(const Task& t);

    // Tasks actually executed by this Runtime's own workers. Distinct from
    // metrics().completed(), which also counts tasks this node handed to a remote
    // thief — summing that across nodes would count stolen tasks twice.
    uint64_t local_executed() const {
        return local_executed_.load(std::memory_order_relaxed);
    }

    void print_steal_stats() const;
    void dump_metrics(const std::string& path) const;

    const Metrics& metrics() const { return metrics_; }
    Metrics&       metrics()       { return metrics_; }

    // Per-task latency rows for external analysis. Call after shutdown().
    void dump_latency_csv(const std::string& path) const { metrics_.dump_csv(path); }

private:
    // Time, run, and account for a task executed by a thread helping out from inside
    // Future::get() rather than by a worker's own dispatch loop.
    void execute_helped(Task& t, bool stolen);

    // Shared completion accounting for both locally and remotely executed tasks.
    void finish_one();

    //WorkDeque                            queue_;
    std::vector<std::unique_ptr<WorkDeque>>       queues;

    // One injection queue per worker so bulk external submission (e.g. a benchmark
    // harness calling submit() in a loop from main) still fans out N-way instead of
    // funneling through a single shared mutex. Round-robin on submit(); each worker
    // checks its own slot first but any worker may drain any slot (all mutex-protected,
    // safe for any caller) so work never gets stuck behind an idle owner.
    std::vector<std::unique_ptr<InjectionQueue>> injection_queues_;
    std::atomic<uint32_t>                        injection_rr_{0};

    // Portable tasks: runnable by local workers, stealable by remote nodes. Kept
    // separate from the per-worker Chase-Lev deques because a remote steal is served
    // by a network thread, and Chase-Lev pop() is owner-only — a network thread must
    // never touch a worker's deque.
    //
    // One pool per task type so a steal can select by capability without scanning:
    // "give me inference work" is an O(1) lookup rather than a search through a
    // single mixed queue.
    std::array<std::unique_ptr<InjectionQueue>, static_cast<std::size_t>(TaskType::Count)>
                          remote_pools_;
    std::atomic<uint32_t> remote_rr_{0};
    std::atomic<uint32_t> node_id_{0};

    Metrics                              metrics_;
    std::vector<std::unique_ptr<Worker>> workers_;

    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> next_id_{1};
    std::atomic<bool>     shutdown_flag_{false};
    std::atomic<uint64_t> spawned_{0};
    std::atomic<uint64_t> local_executed_{0};
    std::atomic<uint64_t> helped_{0};
    std::function<void(const Task&)> task_observer_;
    std::mutex              cv_mutex_;
    std::condition_variable cv_;
};
