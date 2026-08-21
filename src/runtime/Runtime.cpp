#include "runtime/Runtime.hpp"
#include "runtime/TaskRegistry.hpp"
#include <cinttypes>
#include <cstdio>
#include <fstream>

Runtime::Runtime(int num_workers, bool enable_tracing)
    : metrics_(num_workers)
{
    metrics_.enable_tracing(enable_tracing);

    for (int i = 0; i < num_workers; ++i) {
        // Worker needs: its id, the shared queue, a shutdown flag, and a
        // callback to fire when it completes a task.
        // Worker never includes Runtime.hpp — this lambda is the only link back.

        queues.push_back(std::make_unique<WorkDeque>());
        injection_queues_.push_back(std::make_unique<InjectionQueue>());
    }
    std::vector<WorkDeque*> all_queues;
    for (auto& q : queues) {
        all_queues.push_back(q.get());
    }
    std::vector<InjectionQueue*> all_injection;
    for (auto& iq : injection_queues_) {
        all_injection.push_back(iq.get());
    }
    for (int i = 0; i < num_workers; ++i) {
        WorkerContext ctx;
        ctx.all_queues    = all_queues;
        ctx.all_injection = all_injection;
        ctx.remote_pool   = &remote_pool_;
        ctx.metrics       = &metrics_;
        ctx.shutdown_flag = &shutdown_flag_;
        ctx.submitted     = &submitted_;
        ctx.completed     = &completed_;
        ctx.on_complete   = [this]() { on_task_complete(); };

        workers_.push_back(std::make_unique<Worker>(
            i, *queues[i], *injection_queues_[i], std::move(ctx)));
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

    // Stamped before the push so queue wait covers the full time the task sat
    // enqueued, including any time spent waiting to be stolen.
    if (metrics_.tracing_enabled()) {
        t.submit_ns = metrics_.now_ns();
    }

    // Chase-Lev's WorkDeque::push() is owner-only. If we're running on a worker
    // thread (e.g. a recursive spawn() from inside a running task), push onto that
    // worker's own local queue — safe, and better locality than round-robin. Anything
    // submitted from outside a worker thread (main/test/bench code) has no queue it
    // can safely own, so it round-robins across the injection queues instead — bulk
    // external submission (e.g. a benchmark loop) still fans out N-way this way,
    // rather than serializing through one shared queue.
    if (t_current_queue != nullptr) {
        t_current_queue->push(std::move(t));
    } else {
        uint32_t idx = injection_rr_.fetch_add(1, std::memory_order_relaxed) %
                       static_cast<uint32_t>(injection_queues_.size());
        injection_queues_[idx]->push(std::move(t));
    }
}

void Runtime::submit_portable(TaskType type, std::vector<uint8_t> payload, uint32_t cost_hint) {
    Task t;
    t.task_id     = next_id_++;
    t.type        = type;
    t.payload     = std::move(payload);
    t.cost_hint   = cost_hint;
    t.origin_node = node_id_.load(std::memory_order_relaxed);

    submitted_.fetch_add(1, std::memory_order_relaxed);
    metrics_.record_submitted();
    if (metrics_.tracing_enabled()) {
        t.submit_ns = metrics_.now_ns();
    }

    remote_pool_.push(std::move(t));
}

bool Runtime::take_portable(Task& out) {
    return remote_pool_.pop(out);
}

void Runtime::on_remote_task_complete() {
    // The origin counted this task as submitted, so it must count it as completed —
    // but not as locally executed, since another node did the work.
    finish_one();
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
    local_executed_.fetch_add(1, std::memory_order_relaxed);
    finish_one();
}

void Runtime::finish_one() {
    metrics_.record_completed();
    uint64_t done = completed_.fetch_add(1, std::memory_order_relaxed) + 1;

    if (done >= submitted_.load(std::memory_order_relaxed)) {
        cv_.notify_all();
    }
}

void Runtime::execute_helped(Task& t, bool stolen) {
    if (metrics_.tracing_enabled()) {
        uint64_t start = metrics_.now_ns();
        run_task(t);
        uint64_t end = metrics_.now_ns();
        metrics_.record_task(t.task_id, t.submit_ns, start, end, stolen);
    } else {
        run_task(t);
    }
    on_task_complete();
    helped_.fetch_add(1, std::memory_order_relaxed);
}

bool Runtime::try_execute_one(){
    // Called by a thread blocked in Future::get(), so it can help drain pending work
    // instead of idling. WorkDeque::pop() is owner-only — only safe on the calling
    // worker's own queue (if any) — everything else must go through steal() or the
    // injection queue, both of which tolerate any caller thread.
    Task t;

    if (t_current_queue != nullptr && t_current_queue->pop(t)) {
        execute_helped(t, /*stolen=*/false);
        return true;
    }

    for (auto& iq : injection_queues_) {
        if (iq->pop(t)) {
            execute_helped(t, /*stolen=*/true);
            return true;
        }
    }

    if (remote_pool_.pop(t)) {
        execute_helped(t, /*stolen=*/false);
        return true;
    }

    for (auto& q : queues) {
        if (q.get() == t_current_queue) continue;
        if (q->steal(t)) {
            execute_helped(t, /*stolen=*/true);
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

    LatencySummary lat = metrics_.summarize();
    if (lat.total.count > 0) {
        std::fprintf(f, "samples=%" PRIu64 "\n", lat.total.count);
        // Latencies in microseconds; queue_wait is the scheduling-quality signal.
        std::fprintf(f, "queue_wait_us p50=%.1f p95=%.1f p99=%.1f max=%.1f\n",
            lat.queue_wait.p50 / 1000.0, lat.queue_wait.p95 / 1000.0,
            lat.queue_wait.p99 / 1000.0, lat.queue_wait.max / 1000.0);
        std::fprintf(f, "exec_us       p50=%.1f p95=%.1f p99=%.1f max=%.1f\n",
            lat.exec.p50 / 1000.0, lat.exec.p95 / 1000.0,
            lat.exec.p99 / 1000.0, lat.exec.max / 1000.0);
        std::fprintf(f, "total_us      p50=%.1f p95=%.1f p99=%.1f max=%.1f\n",
            lat.total.p50 / 1000.0, lat.total.p95 / 1000.0,
            lat.total.p99 / 1000.0, lat.total.max / 1000.0);
    }
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
