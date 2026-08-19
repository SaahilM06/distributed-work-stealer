#pragma once

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
    void on_task_complete();

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

    //WorkDeque                            queue_;
    std::vector<std::unique_ptr<WorkDeque>>       queues;

    // One injection queue per worker so bulk external submission (e.g. a benchmark
    // harness calling submit() in a loop from main) still fans out N-way instead of
    // funneling through a single shared mutex. Round-robin on submit(); each worker
    // checks its own slot first but any worker may drain any slot (all mutex-protected,
    // safe for any caller) so work never gets stuck behind an idle owner.
    std::vector<std::unique_ptr<InjectionQueue>> injection_queues_;
    std::atomic<uint32_t>                        injection_rr_{0};

    Metrics                              metrics_;
    std::vector<std::unique_ptr<Worker>> workers_;

    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> next_id_{1};
    std::atomic<bool>     shutdown_flag_{false};
    std::atomic<uint64_t> spawned_{0};
    std::atomic<uint64_t> helped_{0};
    std::mutex              cv_mutex_;
    std::condition_variable cv_;
};
