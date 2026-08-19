#pragma once

#include <cstddef>
#include <deque>
#include <mutex>

#include "runtime/Task.hpp"

// Multi-producer, multi-consumer queue for tasks entering the runtime from outside any
// worker thread (e.g. the initial Runtime::submit() calls from main/test/bench code).
// Chase-Lev's per-worker WorkDeque only tolerates a single owner thread calling
// push()/pop() — anything from an external thread has to land somewhere every worker
// can safely drain from, hence this separate mutex-protected queue.
class InjectionQueue {
public:
    void        push(Task task);
    bool        pop(Task& out);
    std::size_t size() const;

private:
    std::deque<Task>   deque_;
    mutable std::mutex mutex_;
};
