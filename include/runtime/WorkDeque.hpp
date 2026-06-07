#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include "runtime/Task.hpp"

class WorkDeque {
private:
    std::deque<Task>   deque_;
    mutable std::mutex mutex_;

public:
    void        push(Task task);
    bool        pop(Task& out);
    std::size_t size() const;
    bool        steal(Task& task);
};
