#include "runtime/Runtime.hpp"
#include "metrics/Metrics.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>

// Burn some CPU so tasks aren't trivially instant.
// Volatile prevents the compiler from optimizing the loop away.
static void synthetic_work(int iterations) {
    volatile int x = 0;
    for (int i = 0; i < iterations; ++i) x += i;
}

static void run_bench(int num_workers, int num_tasks, int work_iterations) {
    std::atomic<int> completed{0};

    Runtime rt(num_workers);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_tasks; ++i) {
        rt.submit([&completed, work_iterations]() {
            synthetic_work(work_iterations);
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    rt.wait_all();

    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double tasks_per_sec = (num_tasks / elapsed_ms) * 1000.0;

    rt.shutdown();

    std::printf("workers=%-2d  tasks=%-6d  work=%-5d  time=%.1f ms  throughput=%.0f tasks/sec\n",
                num_workers, num_tasks, work_iterations, elapsed_ms, tasks_per_sec);
    rt.print_steal_stats();
    rt.dump_metrics("results/metrics.txt");
}

int main() {
    constexpr int NUM_TASKS = 10000;
    constexpr int WORK      = 1000;

    std::printf("=== HydraRT Phase 1 Baseline Benchmark ===\n");
    std::printf("tasks=%d  work_iterations=%d\n\n", NUM_TASKS, WORK);

    for (int workers : {1, 2, 4, 8}) {
        run_bench(workers, NUM_TASKS, WORK);
    }

    return 0;
}
