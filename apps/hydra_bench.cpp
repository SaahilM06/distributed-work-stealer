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

// Uniform workload: every task costs the same. Returns elapsed milliseconds.
static double run_uniform(int num_workers, int num_tasks, int work_iterations,
                          bool trace, bool quiet) {
    Runtime rt(num_workers, trace);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_tasks; ++i) {
        rt.submit([work_iterations]() { synthetic_work(work_iterations); });
    }
    rt.wait_all();
    auto end = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    rt.shutdown();

    if (!quiet) {
        std::printf("workers=%-2d  tasks=%-6d  work=%-5d  time=%.1f ms  throughput=%.0f tasks/sec%s\n",
                    num_workers, num_tasks, work_iterations, elapsed_ms,
                    (num_tasks / elapsed_ms) * 1000.0,
                    trace ? "  [traced]" : "");
        if (trace) {
            rt.metrics().print_summary();
            char path[256];
            std::snprintf(path, sizeof(path), "results/phase6_latency_uniform_w%d.csv", num_workers);
            rt.dump_latency_csv(path);
            rt.dump_metrics("results/metrics.txt");
        }
    }
    return elapsed_ms;
}

// Skewed workload: 10% of tasks cost 20x the rest. This is the shape that makes tail
// latency worth measuring — a mean hides it, p99 does not — and it previews the
// unpredictable per-request cost the ML inference engine (Phase 11) will produce.
static void run_skewed(int num_workers, int num_tasks, int work_iterations) {
    Runtime rt(num_workers, /*enable_tracing=*/true);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_tasks; ++i) {
        int cost = (i % 10 == 0) ? work_iterations * 20 : work_iterations;
        rt.submit([cost]() { synthetic_work(cost); });
    }
    rt.wait_all();
    auto end = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    rt.shutdown();

    std::printf("workers=%-2d  tasks=%-6d  skewed(10%% @ 20x)  time=%.1f ms  throughput=%.0f tasks/sec\n",
                num_workers, num_tasks, elapsed_ms, (num_tasks / elapsed_ms) * 1000.0);
    rt.metrics().print_summary();
    rt.print_steal_stats();

    char path[256];
    std::snprintf(path, sizeof(path), "results/phase6_latency_skewed_w%d.csv", num_workers);
    rt.dump_latency_csv(path);
}

int main() {
    constexpr int NUM_TASKS = 10000;
    constexpr int WORK      = 1000;
    constexpr int REPS      = 5;

    std::printf("=== HydraRT Phase 6: Metrics & Tracing ===\n");
    std::printf("tasks=%d  work_iterations=%d  reps=%d (best of)\n\n", NUM_TASKS, WORK, REPS);

    // Warm-up, discarded. Without it the first configuration measured absorbs all the
    // cold-start cost (page faults, CPU frequency ramp) and whatever runs first looks
    // artificially slow — which made an earlier version of this benchmark report
    // *negative* tracing overhead.
    run_uniform(1, NUM_TASKS, WORK, /*trace=*/false, /*quiet=*/true);
    run_uniform(8, NUM_TASKS, WORK, /*trace=*/true,  /*quiet=*/true);

    std::printf("--- uniform workload: throughput, tracing OFF vs ON ---\n");
    for (int workers : {1, 2, 4, 8}) {
        // Best-of-N for both arms, measured back to back at the same worker count, so
        // the two numbers see the same machine conditions.
        double best_off = 1e300, best_on = 1e300;
        for (int r = 0; r < REPS; ++r) {
            double off = run_uniform(workers, NUM_TASKS, WORK, false, /*quiet=*/true);
            double on  = run_uniform(workers, NUM_TASKS, WORK, true,  /*quiet=*/true);
            if (off < best_off) best_off = off;
            if (on  < best_on)  best_on  = on;
        }
        std::printf("workers=%-2d  off=%.1f ms (%.0f tasks/sec)   on=%.1f ms (%.0f tasks/sec)"
                    "   tracing cost=%+.1f%%\n",
                    workers,
                    best_off, (NUM_TASKS / best_off) * 1000.0,
                    best_on,  (NUM_TASKS / best_on)  * 1000.0,
                    ((best_on - best_off) / best_off) * 100.0);
    }
    std::printf("(run-to-run variance is roughly +/-5%%, so small negative tracing costs\n"
                " are noise, not a speedup; the 4- and 8-worker figures are the signal.)\n");

    // One traced run per worker count for the latency detail + CSV dump.
    std::printf("\n--- uniform workload: latency detail (tracing ON) ---\n");
    for (int workers : {1, 4, 8}) {
        run_uniform(workers, NUM_TASKS, WORK, /*trace=*/true, /*quiet=*/false);
    }

    std::printf("\n--- skewed workload (tail latency), tracing ON ---\n");
    for (int workers : {1, 4, 8}) {
        run_skewed(workers, NUM_TASKS, WORK);
    }

    return 0;
}
