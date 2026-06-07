#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "runtime/Task.hpp"
#include "runtime/WorkDeque.hpp"

template<typename T>
class Future {
public:

    Future(std::function<bool()> try_work)
        : try_work_(std::move(try_work))
    {}
    // called by the task when it has a result
    void set(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            value_ = value; 
            ready_ = true;
        }
        cv_.notify_all();
    }

    // called by whoever is waiting — blocks until ready
    T get() {
        while(!ready_.load(std::memory_order_acquire)) {
            if(!try_work_()) {
                std::this_thread::yield();
            }
        }
        // std::unique_lock<std::mutex> lock(mutex_);
        // cv_.wait(lock, [this]() {
        //     return ready_.load();
        // });
        return value_;
    }

private:
    T                       value_;
    std::atomic<bool>       ready_{false};
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::function<bool()> try_work_;
};
