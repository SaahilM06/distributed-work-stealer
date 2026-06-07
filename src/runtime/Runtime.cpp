#include "runtime/Runtime.hpp"
#include <cinttypes>
#include <cstdio>
#include <fstream>

Runtime::Runtime(int num_workers) {
    for (int i = 0; i < num_workers; ++i) {
        // Worker needs: its id, the shared queue, a shutdown flag, and a
        // callback to fire when it completes a task.
        // Worker never includes Runtime.hpp — this lambda is the only link back.
        
        queues.push_back(std::make_unique<WorkDeque>());        
    }
    std::vector<WorkDeque*> all_queues;
    for (auto& q : queues) {
        all_queues.push_back(q.get());
    }
    for (int i = 0; i < num_workers; ++i) {
        workers_.push_back(std::make_unique<Worker>(
            i,
            *queues[i],
            all_queues,
            shutdown_flag_,
            [this]() { on_task_complete(); }
        ));
    }

    for (auto& w : workers_) {
        w->start();
    }
}

Runtime::~Runtime() {
    shutdown();
}

void Runtime::submit(std::function<void()> fn) {
    Task t;
    t.task_id = next_id_++;
    t.fn      = std::move(fn);

    // increment submitted BEFORE pushing so completed can never overtake it
    submitted_.fetch_add(1, std::memory_order_relaxed);
    metrics_.record_submitted();
    int idx = worker_num.fetch_add(1, std::memory_order_relaxed) % static_cast<int>(queues.size());
    queues[idx] -> push(std::move(t));
}

void Runtime::wait_all() {
    std::unique_lock<std::mutex> lock(cv_mutex_);
    cv_.wait(lock, [this]() {
        return submitted_.load(std::memory_order_relaxed) > 0 &&
               completed_.load(std::memory_order_relaxed) >=
               submitted_.load(std::memory_order_relaxed);
    });
}

void Runtime::on_task_complete() {
    metrics_.record_completed();
    uint64_t done = completed_.fetch_add(1, std::memory_order_relaxed) + 1;

    if (done >= submitted_.load(std::memory_order_relaxed)) {
        cv_.notify_all();
    }
}

bool Runtime::try_execute_one(){
    for(auto& q: queues) {
        Task t;
        if(q -> pop(t)) {
            t.fn();
            on_task_complete();
            helped_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

void Runtime::dump_metrics(const std::string& path) const {
    uint64_t total_steal_attempts  = 0;
    uint64_t total_steal_successes = 0;
    for (int i = 0; i < (int)workers_.size(); ++i) {
        total_steal_attempts  += workers_[i]->steal_attempts();
        total_steal_successes += workers_[i]->steal_successes();
    }

    std::FILE* f = std::fopen(path.c_str(), "a");
    if (!f) return;
    std::fprintf(f, "submitted=%" PRIu64 "\n",  submitted_.load(std::memory_order_relaxed));
    std::fprintf(f, "completed=%" PRIu64 "\n",  completed_.load(std::memory_order_relaxed));
    std::fprintf(f, "spawned=%" PRIu64 "\n",    spawned_.load(std::memory_order_relaxed));
    std::fprintf(f, "helped=%" PRIu64 "\n",     helped_.load(std::memory_order_relaxed));
    std::fprintf(f, "steal_attempts=%" PRIu64 "\n",  total_steal_attempts);
    std::fprintf(f, "steal_successes=%" PRIu64 "\n", total_steal_successes);
    std::fprintf(f, "steal_rate=%.1f%%\n",
        total_steal_attempts > 0 ? (total_steal_successes * 100.0 / total_steal_attempts) : 0.0);
    std::fprintf(f, "---\n");
    std::fclose(f);
}



void Runtime::print_steal_stats() const {
    uint64_t total_attempts  = 0;
    uint64_t total_successes = 0;
    for (int i = 0; i < (int)workers_.size(); ++i) {
        uint64_t a = workers_[i]->steal_attempts();
        uint64_t s = workers_[i]->steal_successes();
        total_attempts  += a;
        total_successes += s;
        std::printf("  worker %d: attempts=%" PRIu64 "  successes=%" PRIu64 "  rate=%.1f%%\n",
                    i, a, s, a > 0 ? (s * 100.0 / a) : 0.0);
    }
    std::printf("  total:    attempts=%" PRIu64 "  successes=%" PRIu64 "  rate=%.1f%%\n",
                total_attempts, total_successes,
                total_attempts > 0 ? (total_successes * 100.0 / total_attempts) : 0.0);
}

void Runtime::shutdown() {
    shutdown_flag_.store(true, std::memory_order_relaxed);
    for (auto& w : workers_) {
        w->join();
    }
}
