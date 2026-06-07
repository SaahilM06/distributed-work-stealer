#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "runtime/WorkDeque.hpp"

class Worker {
public:
    Worker(int id,
           WorkDeque& queue,
           std::vector<WorkDeque*> all_queues,
           std::atomic<bool>& shutdown_flag,
           std::function<void()> on_complete);

    void start();
    void join();

    uint64_t steal_attempts() const { return steal_attempts_.load(std::memory_order_relaxed); }
    uint64_t steal_successes() const { return steal_successes_.load(std::memory_order_relaxed); }

private:
    void run();

    int                    id_;
    WorkDeque&             queue_;
    std::vector<WorkDeque*> all_queues_;
    std::atomic<bool>&     shutdown_flag_;
    std::function<void()>  on_complete_;
    std::thread            thread_;

    std::atomic<uint64_t>  steal_attempts_{0};
    std::atomic<uint64_t>  steal_successes_{0};
};
