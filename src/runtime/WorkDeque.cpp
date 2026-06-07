#include "runtime/WorkDeque.hpp"

void WorkDeque::push(Task task) {
    std::lock_guard<std::mutex> lock(mutex_);
    deque_.push_back(std::move(task));

}

bool WorkDeque::pop(Task& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(deque_.size() == 0) {
        return false;
    }
    out = std::move(deque_.front());
    deque_.pop_front();
    return true;
}

std::size_t WorkDeque::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return deque_.size();
}

//steal function that takes from the back (thief thread will take from the back)
bool WorkDeque::steal(Task& out) {
     std::lock_guard<std::mutex> lock(mutex_);
     if(deque_.size() == 0) {
        return false;
     }

     out = std::move(deque_.back());
     deque_.pop_back(); 
     return true;
}
