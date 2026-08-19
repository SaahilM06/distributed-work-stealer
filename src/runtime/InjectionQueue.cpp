#include "runtime/InjectionQueue.hpp"

void InjectionQueue::push(Task task) {
    std::lock_guard<std::mutex> lock(mutex_);
    deque_.push_back(std::move(task));
}

bool InjectionQueue::pop(Task& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (deque_.empty()) {
        return false;
    }
    out = std::move(deque_.front());
    deque_.pop_front();
    return true;
}

std::size_t InjectionQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return deque_.size();
}
