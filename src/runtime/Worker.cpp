#include "runtime/Worker.hpp"
#include <thread>

Worker::Worker(int id,
               WorkDeque& queue,
               std::atomic<bool>& shutdown_flag,
               std::function<void()> on_complete)
    : id_(id)
    , queue_(queue)
    , shutdown_flag_(shutdown_flag)
    , on_complete_(std::move(on_complete))
{}

void Worker::start() {
    // TODO: launch thread_ running run()
    thread_ = std::thread(&Worker::run, this);
}

void Worker::join() {
    // TODO: join thread_ if it's joinable
    if(thread_.joinable()) {
        thread_.join();
    }
}

void Worker::run() {
    while (true) {
        if (shutdown_flag_.load(std::memory_order_relaxed) && queue_.size() == 0) {
            break;
        }

        Task t;
        if (queue_.pop(t)) {
            t.fn();
            on_complete_();
        } else {
            std::this_thread::yield();
        }
    }
}
