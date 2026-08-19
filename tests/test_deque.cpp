#include "runtime/WorkDeque.hpp"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

// Owner pop() is LIFO (Chase-Lev semantics — see ChaseLevDeque.hpp): it returns the
// most recently pushed task, taking from the same end push() writes to.
static void test_push_pop() {
    WorkDeque q;
    Task t1; t1.task_id = 1;
    Task t2; t2.task_id = 2;

    q.push(t1);
    q.push(t2);
    assert(q.size() == 2);

    Task out;
    assert(q.pop(out) && out.task_id == 2);
    assert(q.pop(out) && out.task_id == 1);
    assert(q.size() == 0);
    std::printf("PASS test_push_pop\n");
}

static void test_pop_empty() {
    WorkDeque q;
    Task out;
    assert(!q.pop(out));
    std::printf("PASS test_pop_empty\n");
}

// steal() is FIFO relative to push order: the oldest unstolen task first.
static void test_steal_order() {
    WorkDeque q;
    for (uint64_t i = 1; i <= 3; ++i) {
        Task t; t.task_id = i;
        q.push(t);
    }

    Task out;
    assert(q.steal(out) && out.task_id == 1);
    assert(q.steal(out) && out.task_id == 2);
    assert(q.steal(out) && out.task_id == 3);
    assert(!q.steal(out));
    std::printf("PASS test_steal_order\n");
}

// Push well past the default initial capacity to force grow() to run more than once,
// then drain everything back out and check nothing was lost or corrupted.
static void test_growth() {
    constexpr int N = 5000; // default initial capacity is 1024
    WorkDeque q;
    for (uint64_t i = 0; i < (uint64_t)N; ++i) {
        Task t; t.task_id = i;
        q.push(t);
    }
    assert(q.size() == (std::size_t)N);

    std::vector<bool> seen(N, false);
    Task out;
    int count = 0;
    while (q.pop(out)) {
        assert(!seen[out.task_id]);
        seen[out.task_id] = true;
        ++count;
    }
    assert(count == N);
    std::printf("PASS test_growth: %d tasks survived grow()\n", N);
}

// One owner thread pushes N tasks, then the owner (via pop) and several thief threads
// (via steal) race to drain the deque concurrently. Every task must be observed exactly
// once — no duplicates, no losses — which is the core correctness property a Chase-Lev
// deque has to hold under real concurrent stealing.
static void test_concurrent_owner_and_thieves() {
    constexpr int N            = 50000;
    constexpr int NUM_THIEVES  = 4;

    WorkDeque q;
    for (uint64_t i = 0; i < (uint64_t)N; ++i) {
        Task t; t.task_id = i;
        q.push(t);
    }

    std::vector<std::atomic<bool>> seen(N);
    for (auto& s : seen) s.store(false);
    std::atomic<int>  collected{0};
    std::atomic<bool> duplicate{false};
    std::atomic<bool> out_of_range{false};

    auto record = [&](uint64_t id) {
        if (id >= (uint64_t)N) { out_of_range.store(true); return; }
        bool expected = false;
        if (!seen[id].compare_exchange_strong(expected, true)) {
            duplicate.store(true);
        } else {
            collected.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> thieves;
    for (int i = 0; i < NUM_THIEVES; ++i) {
        thieves.emplace_back([&]() {
            Task t;
            while (collected.load(std::memory_order_relaxed) < N) {
                if (q.steal(t)) {
                    record(t.task_id);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    Task t;
    while (collected.load(std::memory_order_relaxed) < N) {
        if (q.pop(t)) {
            record(t.task_id);
        } else {
            std::this_thread::yield();
        }
    }

    for (auto& th : thieves) th.join();

    assert(!out_of_range.load());
    assert(!duplicate.load());
    assert(collected.load() == N);
    assert(q.size() == 0);
    std::printf("PASS test_concurrent_owner_and_thieves: %d tasks, no duplicates/losses\n", N);
}

int main() {
    test_push_pop();
    test_pop_empty();
    test_steal_order();
    test_growth();
    test_concurrent_owner_and_thieves();
    std::printf("All tests passed.\n");
    return 0;
}
