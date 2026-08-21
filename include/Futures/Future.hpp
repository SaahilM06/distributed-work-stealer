#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

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
        std::vector<std::function<void(const T&)>> to_run;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            value_ = std::move(value);
            ready_.store(true, std::memory_order_release);
            // Take the callbacks out under the lock but run them outside it: a
            // continuation may submit more work, and holding this future's lock while
            // doing so invites lock-order problems.
            to_run.swap(continuations_);
        }
        cv_.notify_all();
        for (auto& fn : to_run) fn(value_);
    }

    // called by whoever is waiting — blocks until ready
    T get() {
        while(!ready_.load(std::memory_order_acquire)) {
            if(!try_work_()) {
                std::this_thread::yield();
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }

    bool ready() const { return ready_.load(std::memory_order_acquire); }

    // Run `fn` when this future resolves, without blocking a thread in the meantime.
    //
    // This is what a request pipeline needs and get() cannot give: a pipeline is a
    // chain (decode -> preprocess -> infer -> postprocess), and using get() between
    // stages parks a whole worker on every in-flight request. With then(), a stage
    // that isn't ready costs nothing — the continuation is simply invoked later, by
    // whichever thread completes the previous stage.
    void then(std::function<void(const T&)> fn) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!ready_.load(std::memory_order_acquire)) {
                continuations_.push_back(std::move(fn));
                return;
            }
        }
        // Already resolved — run immediately rather than queueing a callback that
        // nothing would ever fire.
        fn(value_);
    }

private:
    T                       value_{};
    std::atomic<bool>       ready_{false};
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::function<bool()>   try_work_;
    std::vector<std::function<void(const T&)>> continuations_;
};
