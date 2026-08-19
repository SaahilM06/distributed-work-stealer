#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "runtime/Task.hpp"

// Classic Chase-Lev lock-free work-stealing deque.
//
// Ownership discipline (this is what makes it lock-free, and violating it is a data
// race): push()/pop() may only ever be called by the single owning thread. Any other
// thread must use steal(). push()/pop() operate on `bottom_` (LIFO — pop returns the
// most recently pushed task, which is what you want for cache locality in recursive
// divide-and-conquer task graphs). steal() operates on `top_` via CAS, and always takes
// the oldest unstolen task (FIFO relative to push order), which amortizes better for
// thieves picking up coarse-grained work.
class ChaseLevDeque {
public:
    explicit ChaseLevDeque(std::size_t initial_capacity = 1024)
        : buffer_(new RingBuffer(next_pow2(initial_capacity))) {}

    ~ChaseLevDeque() {
        RingBuffer* buf = buffer_.load(std::memory_order_relaxed);
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        for (int64_t i = t; i < b; ++i) {
            delete buf->get(i);
        }
        delete buf;
    }

    ChaseLevDeque(const ChaseLevDeque&)            = delete;
    ChaseLevDeque& operator=(const ChaseLevDeque&) = delete;

    // Owner-only.
    void push(Task task) {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_acquire);
        RingBuffer* buf = buffer_.load(std::memory_order_relaxed);

        if (b - t >= buf->capacity - 1) {
            RingBuffer* grown = buf->grow(t, b);
            old_buffers_.emplace_back(buf);
            buffer_.store(grown, std::memory_order_release);
            buf = grown;
        }

        // A direct release store here (rather than a separate release fence before a
        // relaxed store) is what makes this task's construction visible to a thief's
        // paired acquire load of bottom_ in steal() — deliberately not relying on the
        // fence+relaxed-store idiom for this specific edge, since it's less obviously
        // correct to readers and to tools that reason about lock-free code.
        buf->put(b, new Task(std::move(task)));
        bottom_.store(b + 1, std::memory_order_release);
    }

    // Owner-only. LIFO: returns the most recently pushed task.
    bool pop(Task& out) {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        RingBuffer* buf = buffer_.load(std::memory_order_relaxed);
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t = top_.load(std::memory_order_relaxed);

        if (t > b) {
            // Deque was already empty; restore bottom_.
            bottom_.store(b + 1, std::memory_order_relaxed);
            return false;
        }

        Task* task_ptr = buf->get(b);
        if (t == b) {
            // Last element — racing against thieves for it.
            if (!top_.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                // Lost the race; a thief took it.
                bottom_.store(b + 1, std::memory_order_relaxed);
                return false;
            }
            bottom_.store(b + 1, std::memory_order_relaxed);
        }

        out = std::move(*task_ptr);
        delete task_ptr;
        return true;
    }

    // Any thread. FIFO relative to push order: returns the oldest unstolen task.
    bool steal(Task& out) {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t >= b) {
            return false;
        }

        RingBuffer* buf = buffer_.load(std::memory_order_acquire);
        Task* task_ptr = buf->get(t);

        if (!top_.compare_exchange_strong(t, t + 1,
                std::memory_order_seq_cst, std::memory_order_relaxed)) {
            // Another thief (or the owner, via pop) got it first.
            return false;
        }

        out = std::move(*task_ptr);
        delete task_ptr;
        return true;
    }

    std::size_t size() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b > t ? static_cast<std::size_t>(b - t) : 0;
    }

private:
    struct RingBuffer {
        int64_t capacity;
        int64_t mask;
        std::unique_ptr<std::atomic<Task*>[]> slots;

        explicit RingBuffer(int64_t cap)
            : capacity(cap), mask(cap - 1), slots(new std::atomic<Task*>[static_cast<std::size_t>(cap)]) {}

        Task* get(int64_t idx) const {
            return slots[static_cast<std::size_t>(idx & mask)].load(std::memory_order_relaxed);
        }

        void put(int64_t idx, Task* task) {
            slots[static_cast<std::size_t>(idx & mask)].store(task, std::memory_order_relaxed);
        }

        // Owner-only. Old buffer is kept alive by the caller (not freed here) since
        // thieves may still be mid-read against it via a pointer they loaded earlier.
        RingBuffer* grow(int64_t top, int64_t bottom) const {
            auto* grown = new RingBuffer(capacity * 2);
            for (int64_t i = top; i < bottom; ++i) {
                grown->put(i, get(i));
            }
            return grown;
        }
    };

    static int64_t next_pow2(std::size_t n) {
        int64_t p = 1;
        while (static_cast<std::size_t>(p) < n) p <<= 1;
        return p;
    }

    alignas(64) std::atomic<int64_t> top_{0};
    alignas(64) std::atomic<int64_t> bottom_{0};
    std::atomic<RingBuffer*> buffer_;

    // Retired buffers from grow() — kept alive for the deque's lifetime rather than
    // freed immediately, since a thief may hold a pointer to one mid-steal.
    std::vector<std::unique_ptr<RingBuffer>> old_buffers_;
};
