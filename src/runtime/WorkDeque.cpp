#include "runtime/WorkDeque.hpp"

thread_local WorkDeque* t_current_queue = nullptr;

void WorkDeque::push(Task task) {
    deque_.push(std::move(task));
}

bool WorkDeque::pop(Task& out) {
    return deque_.pop(out);
}

std::size_t WorkDeque::size() const {
    return deque_.size();
}

bool WorkDeque::steal(Task& out) {
    return deque_.steal(out);
}
