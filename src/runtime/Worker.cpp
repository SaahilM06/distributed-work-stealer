#include "runtime/Worker.hpp"
#include <thread>

Worker::Worker(int id,
               WorkDeque& queue,
               std::vector<WorkDeque*> all_queues,
               std::atomic<bool>& shutdown_flag,
               std::function<void()> on_complete)
    : id_(id)
    , queue_(queue)
    , all_queues_(std::move(all_queues))
    , shutdown_flag_(shutdown_flag)
    , on_complete_(std::move(on_complete))
{}

void Worker::start() {
    thread_ = std::thread(&Worker::run, this);
}

void Worker::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Worker::run() {
    while (true) {
        bool all_empty = true;
        for (WorkDeque* q : all_queues_) {
            if (q->size() > 0) { all_empty = false; break; }
        }
        if (shutdown_flag_.load(std::memory_order_relaxed) && all_empty) {
            break;
        }

        Task t;
        if (queue_.pop(t)) {
            t.fn();
            on_complete_();
        } else {
            bool stolen = false;
            for (WorkDeque* q : all_queues_) {
                if (q == &queue_) continue;
                steal_attempts_.fetch_add(1, std::memory_order_relaxed);
                if (q->steal(t)) {
                    steal_successes_.fetch_add(1, std::memory_order_relaxed);
                    t.fn();
                    on_complete_();
                    stolen = true;
                    break;
                }
            }
            if (!stolen) {
                std::this_thread::yield();
            }
        }
    }
}
