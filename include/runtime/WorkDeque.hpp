#pragma once

#include <cstddef>

#include "runtime/ChaseLevDeque.hpp"
#include "runtime/Task.hpp"

// Per-worker local queue. push()/pop() are owner-only (see ChaseLevDeque); steal() is
// safe from any thread. `t_current_queue` lets any code (chiefly Runtime::submit() and
// Runtime::try_execute_one()) tell whether it's running on a worker thread and, if so,
// which queue that worker owns — Worker::run() sets it once at thread start.
class WorkDeque {
public:
    void        push(Task task);
    bool        pop(Task& out);
    std::size_t size() const;
    bool        steal(Task& task);

private:
    ChaseLevDeque deque_;
};

extern thread_local WorkDeque* t_current_queue;
