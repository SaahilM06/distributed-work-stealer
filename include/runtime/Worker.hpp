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

    // Portable tasks that any local worker may run and any remote node may steal,
    // one pool per task type.
    std::vector<InjectionQueue*> remote_pools;

    Metrics*                     metrics       = nullptr;
    std::atomic<bool>*           shutdown_flag = nullptr;
    const std::atomic<uint64_t>* submitted     = nullptr;
    const std::atomic<uint64_t>* completed     = nullptr;
    // Receives the task that just finished. Passing the task (rather than a bare
    // notification) is what lets a multi-stage pipeline submit its next stage — and
    // Runtime submits that next stage BEFORE counting this one complete, so
    // wait_all() can never see a momentarily balanced ledger and return early.
    std::function<void(const Task&)> on_complete;
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
