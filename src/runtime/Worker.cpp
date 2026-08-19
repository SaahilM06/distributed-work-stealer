#include "runtime/Worker.hpp"
#include <thread>

Worker::Worker(int id,
               WorkDeque& queue,
               std::vector<WorkDeque*> all_queues,
               InjectionQueue& own_injection,
               std::vector<InjectionQueue*> all_injection,
               Metrics& metrics,
               std::atomic<bool>& shutdown_flag,
               const std::atomic<uint64_t>& submitted,
               const std::atomic<uint64_t>& completed,
               std::function<void()> on_complete)
    : id_(id)
    , queue_(queue)
    , all_queues_(std::move(all_queues))
    , own_injection_(own_injection)
    , all_injection_(std::move(all_injection))
    , metrics_(metrics)
    , shutdown_flag_(shutdown_flag)
    , submitted_(submitted)
    , completed_(completed)
    , on_complete_(std::move(on_complete))
{}

void Worker::start() {
    thread_ = std::thread(&Worker::run, this);
}

void Worker::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Worker::execute(Task& t, bool stolen) {
    if (metrics_.tracing_enabled()) {
        uint64_t start = metrics_.now_ns();
        t.fn();
        uint64_t end = metrics_.now_ns();
        metrics_.record_task(t.task_id, t.submit_ns, start, end, stolen);
    } else {
        t.fn();
    }
    on_complete_();
}

void Worker::run() {
    t_current_queue = &queue_;
    metrics_.bind_worker_shard(id_);

    while (true) {
        // A task only counts as completed once it has fully executed, so
        // submitted == completed means nothing is queued or running anywhere in the
        // system (local queues, injection queues, or mid-flight) — an O(1) shutdown
        // check that avoids scanning every queue's size() (each injection size() is a
        // mutex lock/unlock) on every idle-spin iteration.
        bool all_done = submitted_.load(std::memory_order_relaxed) ==
                         completed_.load(std::memory_order_relaxed);
        if (shutdown_flag_.load(std::memory_order_relaxed) && all_done) {
            break;
        }

        Task t;
        // size() is a cheap racy hint (two relaxed loads, no fence) — skip the full
        // pop() (which pays for a seq_cst fence even when empty) unless it looks
        // worth trying. pop() stays the source of truth either way.
        if (queue_.size() > 0 && queue_.pop(t)) {
            execute(t, /*stolen=*/false);
            continue;
        }

        if (own_injection_.pop(t)) {
            execute(t, /*stolen=*/false);
            continue;
        }

        // Peer injection queues checked before peer local queues: imbalance in bulk
        // external submission (e.g. a benchmark loop with no recursive spawning) shows
        // up as a lopsided injection queue, not a lopsided local Chase-Lev queue, so
        // this order avoids wasting steal attempts against local queues that are never
        // populated in that workload. Recursive fork-join workloads still fall through
        // to the local-queue steal loop below when injection queues are also empty.
        bool found = false;
        for (InjectionQueue* iq : all_injection_) {
            if (iq == &own_injection_) continue;
            if (iq->pop(t)) {
                execute(t, /*stolen=*/true);
                found = true;
                break;
            }
        }

        if (!found) {
            for (WorkDeque* q : all_queues_) {
                if (q == &queue_) continue;
                steal_attempts_.fetch_add(1, std::memory_order_relaxed);
                if (q->steal(t)) {
                    steal_successes_.fetch_add(1, std::memory_order_relaxed);
                    execute(t, /*stolen=*/true);
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            std::this_thread::yield();
        }
    }

    metrics_.unbind_worker_shard();
    t_current_queue = nullptr;
}
