#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "runtime/Task.hpp"
#include "runtime/WorkDeque.hpp"
#include "runtime/Worker.hpp"
#include "metrics/Metrics.hpp"

class Runtime {
public:
    explicit Runtime(int num_workers);
    ~Runtime();

    // wrap a callable into a Task and push it onto the global queue
    void submit(std::function<void()> fn);

    // block until every submitted task has completed
    void wait_all();

    // signal all workers to stop and join their threads
    void shutdown();

    // called by each worker after it finishes executing a task
    void on_task_complete();

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
    std::mutex              cv_mutex_;
    std::condition_variable cv_;
};
