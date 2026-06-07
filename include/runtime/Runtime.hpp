#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/Task.hpp"
#include "runtime/WorkDeque.hpp"
#include "runtime/Worker.hpp"
#include "metrics/Metrics.hpp"
#include "Futures/Future.hpp"

class Runtime {
public:
    explicit Runtime(int num_workers);
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

private:
    //WorkDeque                            queue_;
    std::vector<std::unique_ptr<WorkDeque>> queues;

    Metrics                              metrics_;
    std::vector<std::unique_ptr<Worker>> workers_;

    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> next_id_{1};
    std::atomic<bool>     shutdown_flag_{false};
    std::atomic<int32_t> worker_num{0};
    std::atomic<uint64_t> spawned_{0};
    std::atomic<uint64_t> helped_{0};
    std::mutex              cv_mutex_;
    std::condition_variable cv_;
};
