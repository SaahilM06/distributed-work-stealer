#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "metrics/Metrics.hpp"
#include "runtime/InjectionQueue.hpp"
#include "runtime/WorkDeque.hpp"

class Worker {
public:
    Worker(int id,
           WorkDeque& queue,
           std::vector<WorkDeque*> all_queues,
           InjectionQueue& own_injection,
           std::vector<InjectionQueue*> all_injection,
           Metrics& metrics,
           std::atomic<bool>& shutdown_flag,
           const std::atomic<uint64_t>& submitted,
           const std::atomic<uint64_t>& completed,
           std::function<void()> on_complete);

    void start();
    void join();

    uint64_t steal_attempts() const { return steal_attempts_.load(std::memory_order_relaxed); }
    uint64_t steal_successes() const { return steal_successes_.load(std::memory_order_relaxed); }

private:
    void run();

    // Time (when tracing is on), run, and account for one task. Every dispatch path in
    // run() goes through here so timing can't drift between them.
    void execute(Task& t, bool stolen);

    int                    id_;
    WorkDeque&             queue_;
    std::vector<WorkDeque*> all_queues_;
    InjectionQueue&        own_injection_;
    std::vector<InjectionQueue*> all_injection_;
    Metrics&               metrics_;
    std::atomic<bool>&     shutdown_flag_;
    const std::atomic<uint64_t>& submitted_;
    const std::atomic<uint64_t>& completed_;
    std::function<void()>  on_complete_;
    std::thread            thread_;

    std::atomic<uint64_t>  steal_attempts_{0};
    std::atomic<uint64_t>  steal_successes_{0};
};
