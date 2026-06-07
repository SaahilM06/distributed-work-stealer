#include "runtime/Runtime.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

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

// recursive merge sort using spawn() + futures
static std::vector<int> merge_sort(Runtime& rt, std::vector<int> data) {
    if (data.size() <= 64) {
        std::sort(data.begin(), data.end());
        return data;
    }

    int mid = data.size() / 2;
    std::vector<int> left(data.begin(), data.begin() + mid);
    std::vector<int> right(data.begin() + mid, data.end());

    auto left_fut  = rt.spawn<std::vector<int>>([&rt, left]()  { return merge_sort(rt, left); });
    auto right_fut = rt.spawn<std::vector<int>>([&rt, right]() { return merge_sort(rt, right); });

    std::vector<int> sorted_left  = left_fut->get();
    std::vector<int> sorted_right = right_fut->get();

    std::vector<int> merged;
    merged.reserve(data.size());
    std::merge(sorted_left.begin(),  sorted_left.end(),
               sorted_right.begin(), sorted_right.end(),
               std::back_inserter(merged));
    return merged;
}

static void test_merge_sort() {
    constexpr int N = 1024;
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) data[i] = N - i;

    Runtime rt(4);
    std::vector<int> result = merge_sort(rt, data);
    rt.wait_all();
    rt.shutdown();

    assert((int)result.size() == N);
    assert(std::is_sorted(result.begin(), result.end()));
    std::printf("PASS test_merge_sort: sorted %d elements\n", N);
    rt.dump_metrics("results/metrics.txt");
}

// parallel fibonacci using spawn() + futures
// base case at n <= 20 to avoid spawning thousands of tiny tasks
static int fib(Runtime& rt, int n) {
    if (n <= 20) {
        if (n <= 1) return n;
        return fib(rt, n - 1) + fib(rt, n - 2);
    }

    auto f1 = rt.spawn<int>([&rt, n]() { return fib(rt, n - 1); });
    auto f2 = rt.spawn<int>([&rt, n]() { return fib(rt, n - 2); });

    return f1->get() + f2->get();
}

static void test_fib() {
    // known fibonacci values to assert against
    // fib(30) = 832040
    constexpr int N        = 30;
    constexpr int EXPECTED = 832040;

    Runtime rt(4);
    int result = fib(rt, N);
    rt.wait_all();
    rt.shutdown();

    assert(result == EXPECTED);
    std::printf("PASS test_fib: fib(%d) = %d\n", N, result);
    rt.dump_metrics("results/metrics.txt");
}

int main() {
    test_basic_execution();
    test_single_worker();
    test_two_batches();
    test_distribution();
    test_merge_sort();
    test_fib();
    std::printf("All tests passed.\n");
    return 0;
}
