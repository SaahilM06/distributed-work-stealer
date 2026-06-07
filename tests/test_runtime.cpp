#include "runtime/Runtime.hpp"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <thread>

// submit N tasks that each increment a counter, assert all completed
static void test_basic_execution() {
    constexpr int NUM_TASKS = 1000;
    std::atomic<int> counter{0};

    Runtime rt(4);
    for (int i = 0; i < NUM_TASKS; ++i) {
        rt.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    rt.wait_all();
    rt.shutdown();

    assert(counter.load() == NUM_TASKS);
    std::printf("PASS test_basic_execution: %d/%d tasks completed\n", counter.load(), NUM_TASKS);
}

// single worker, verify no deadlock or missed tasks
static void test_single_worker() {
    constexpr int NUM_TASKS = 100;
    std::atomic<int> counter{0};

    Runtime rt(1);
    for (int i = 0; i < NUM_TASKS; ++i) {
        rt.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    rt.wait_all();
    rt.shutdown();

    assert(counter.load() == NUM_TASKS);
    std::printf("PASS test_single_worker: %d/%d tasks completed\n", counter.load(), NUM_TASKS);
}

// submit tasks in two batches with wait_all between them
static void test_two_batches() {
    std::atomic<int> counter{0};

    Runtime rt(4);

    for (int i = 0; i < 500; ++i) {
        rt.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    rt.wait_all();
    assert(counter.load() == 500);

    for (int i = 0; i < 500; ++i) {
        rt.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    rt.wait_all();
    rt.shutdown();

    assert(counter.load() == 1000);
    std::printf("PASS test_two_batches: counter=%d\n", counter.load());
}

// verify tasks run on more than one thread and show per-worker counts
static void test_distribution() {
    std::map<std::thread::id, int> task_counts;
    std::mutex map_mutex;

    constexpr int NUM_TASKS = 1000;
    constexpr int NUM_WORKERS = 4;

    Runtime rt(NUM_WORKERS);
    for (int i = 0; i < NUM_TASKS; ++i) {
        rt.submit([&task_counts, &map_mutex]() {
            std::lock_guard<std::mutex> lock(map_mutex);
            task_counts[std::this_thread::get_id()]++;
        });
    }
    rt.wait_all();
    rt.shutdown();

    std::printf("\n--- Distribution Test (%d tasks, %d workers) ---\n", NUM_TASKS, NUM_WORKERS);
    int worker_num = 0;
    for (auto& [tid, count] : task_counts) {
        int pct = (count * 100) / NUM_TASKS;
        std::printf("  worker %d: %4d tasks (%2d%%) ", worker_num++, count, pct);
        for (int i = 0; i < pct; i += 2) std::printf("#");
        std::printf("\n");
    }
    std::printf("  total threads used: %zu\n", task_counts.size());

    assert(task_counts.size() > 1);
    std::printf("PASS test_distribution\n\n");
}

int main() {
    test_basic_execution();
    test_single_worker();
    test_two_batches();
    test_distribution();
    std::printf("All tests passed.\n");
    return 0;
}
