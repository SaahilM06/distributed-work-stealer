#include "runtime/Runtime.hpp"
#include <atomic>
#include <cassert>
#include <cstdio>

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

int main() {
    test_basic_execution();
    test_single_worker();
    test_two_batches();
    std::printf("All tests passed.\n");
    return 0;
}
