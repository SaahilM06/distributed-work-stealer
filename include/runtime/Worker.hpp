#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "metrics/Metrics.hpp"
#include "runtime/InjectionQueue.hpp"
#include "runtime/WorkDeque.hpp"

// Everything a worker needs that is shared across all workers. Bundled into a struct
// because the pieces keep growing (metrics, remote pool, and more coming with the
// adaptive scheduler) and a ten-argument constructor had stopped being readable.
struct WorkerContext {
    std::vector<WorkDeque*>      all_queues;
    std::vector<InjectionQueue*> all_injection;

    // Portable tasks that any local worker may run and any remote node may steal.
    InjectionQueue* remote_pool = nullptr;

    Metrics*                     metrics       = nullptr;
    std::atomic<bool>*           shutdown_flag = nullptr;
    const std::atomic<uint64_t>* submitted     = nullptr;
    const std::atomic<uint64_t>* completed     = nullptr;
    std::function<void()>        on_complete;
};

class Worker {
public:
    Worker(int id, WorkDeque& own_queue, InjectionQueue& own_injection, WorkerContext ctx);

    void start();
    void join();

    uint64_t steal_attempts() const { return steal_attempts_.load(std::memory_order_relaxed); }
    uint64_t steal_successes() const { return steal_successes_.load(std::memory_order_relaxed); }

private:
    void run();

    // Time (when tracing is on), run, and account for one task. Every dispatch path in
    // run() goes through here so timing can't drift between them.
    void execute(Task& t, bool stolen);

    int             id_;
    WorkDeque&      queue_;
    InjectionQueue& own_injection_;
    WorkerContext   ctx_;
    std::thread     thread_;

    std::atomic<uint64_t> steal_attempts_{0};
    std::atomic<uint64_t> steal_successes_{0};
};
