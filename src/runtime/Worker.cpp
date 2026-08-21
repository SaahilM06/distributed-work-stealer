#include "runtime/Worker.hpp"
#include "runtime/TaskRegistry.hpp"
#include <chrono>
#include <thread>

Worker::Worker(int id, WorkDeque& own_queue, InjectionQueue& own_injection, WorkerContext ctx)
    : id_(id)
    , queue_(own_queue)
    , own_injection_(own_injection)
    , ctx_(std::move(ctx))
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
    Metrics& metrics = *ctx_.metrics;
    if (metrics.tracing_enabled()) {
        uint64_t start = metrics.now_ns();
        run_task(t);
        uint64_t end = metrics.now_ns();
        metrics.record_task(t.task_id, t.submit_ns, start, end, stolen);
    } else {
        run_task(t);
    }
    ctx_.on_complete();
}

void Worker::run() {
    t_current_queue = &queue_;
    ctx_.metrics->bind_worker_shard(id_);

    // Consecutive rounds that found no work anywhere. Drives the idle backoff below.
    uint32_t idle_rounds = 0;

    while (true) {
        // A task only counts as completed once it has fully executed, so
        // submitted == completed means nothing is queued or running anywhere in the
        // system (local queues, injection queues, or mid-flight) — an O(1) shutdown
        // check that avoids scanning every queue's size() (each injection size() is a
        // mutex lock/unlock) on every idle-spin iteration.
        bool all_done = ctx_.submitted->load(std::memory_order_relaxed) ==
                        ctx_.completed->load(std::memory_order_relaxed);
        if (ctx_.shutdown_flag->load(std::memory_order_relaxed) && all_done) {
            break;
        }

        Task t;
        // size() is a cheap racy hint (two relaxed loads, no fence) — skip the full
        // pop() (which pays for a seq_cst fence even when empty) unless it looks
        // worth trying. pop() stays the source of truth either way.
        if (queue_.size() > 0 && queue_.pop(t)) {
            idle_rounds = 0;
            execute(t, /*stolen=*/false);
            continue;
        }

        if (own_injection_.pop(t)) {
            idle_rounds = 0;
            execute(t, /*stolen=*/false);
            continue;
        }

        // Portable tasks. Local workers drain this pool too — running work here beats
        // shipping it over a network — so remote nodes only win tasks from it while
        // this node's workers are busy, which is exactly when stealing should happen.
        if (ctx_.remote_pool != nullptr && ctx_.remote_pool->pop(t)) {
            idle_rounds = 0;
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
        for (InjectionQueue* iq : ctx_.all_injection) {
            if (iq == &own_injection_) continue;
            if (iq->pop(t)) {
                execute(t, /*stolen=*/true);
                found = true;
                break;
            }
        }

        if (!found) {
            for (WorkDeque* q : ctx_.all_queues) {
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

        if (found) {
            idle_rounds = 0;
            continue;
        }

        // Idle backoff. Spinning on yield() forever is fine when workers == cores and
        // the process owns the machine, but it is actively harmful once several node
        // processes share hardware: an idle node's workers burn cores that a busy node
        // needs, which is exactly the case remote stealing exists to fix. Escalate to
        // real sleeps so an idle node costs almost nothing.
        ++idle_rounds;
        if (idle_rounds < 64) {
            std::this_thread::yield();
        } else if (idle_rounds < 512) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ctx_.metrics->unbind_worker_shard();
    t_current_queue = nullptr;
}
