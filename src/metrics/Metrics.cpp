#include "metrics/Metrics.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>

thread_local Metrics::Shard* Metrics::t_shard_ = nullptr;

static uint64_t percentile(const std::vector<uint64_t>& sorted, double p) {
    if (sorted.empty()) return 0;
    double      rank = (p / 100.0) * static_cast<double>(sorted.size());
    std::size_t idx  = static_cast<std::size_t>(std::ceil(rank));
    if (idx == 0) idx = 1;
    if (idx > sorted.size()) idx = sorted.size();
    return sorted[idx - 1];
}

LatencyStats compute_stats(std::vector<uint64_t> values) {
    LatencyStats st;
    if (values.empty()) return st;

    std::sort(values.begin(), values.end());
    st.count = values.size();
    st.min   = values.front();
    st.max   = values.back();
    st.p50   = percentile(values, 50.0);
    st.p95   = percentile(values, 95.0);
    st.p99   = percentile(values, 99.0);

    long double sum = 0.0L;
    for (uint64_t v : values) sum += static_cast<long double>(v);
    st.mean = static_cast<double>(sum / static_cast<long double>(values.size()));
    return st;
}

Metrics::Metrics(int num_worker_shards)
    : origin_(std::chrono::steady_clock::now())
{
    for (int i = 0; i < num_worker_shards; ++i) {
        auto shard = std::make_unique<Shard>();
        shard->worker_id = i;
        worker_shards_.push_back(std::move(shard));
    }
}

void Metrics::record_submitted() {
    submitted_.fetch_add(1, std::memory_order_relaxed);
}

void Metrics::record_completed() {
    completed_.fetch_add(1, std::memory_order_relaxed);
}

uint64_t Metrics::submitted() const {
    return submitted_.load(std::memory_order_relaxed);
}

uint64_t Metrics::completed() const {
    return completed_.load(std::memory_order_relaxed);
}

uint64_t Metrics::now_ns() const {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - origin_).count());
}

void Metrics::bind_worker_shard(int worker_id) {
    if (worker_id >= 0 && worker_id < static_cast<int>(worker_shards_.size())) {
        t_shard_ = worker_shards_[static_cast<std::size_t>(worker_id)].get();
    }
}

void Metrics::unbind_worker_shard() {
    t_shard_ = nullptr;
}

void Metrics::record_task(uint64_t task_id, uint64_t submit_ns, uint64_t start_ns,
                          uint64_t end_ns, bool stolen) {
    if (!tracing_.load(std::memory_order_relaxed)) return;

    TaskSample s;
    s.task_id   = task_id;
    s.submit_ns = submit_ns;
    s.start_ns  = start_ns;
    s.end_ns    = end_ns;
    s.stolen    = stolen;

    // Worker threads append to their own shard with no lock. Anything else (a thread
    // helping from inside Future::get()) falls back to the shared external shard.
    if (Shard* shard = t_shard_) {
        s.worker_id = shard->worker_id;
        if (shard->samples.size() < kMaxSamplesPerShard) {
            shard->samples.push_back(s);
        }
        return;
    }

    s.worker_id = -1;
    std::lock_guard<std::mutex> lock(external_mutex_);
    if (external_shard_.samples.size() < kMaxSamplesPerShard) {
        external_shard_.samples.push_back(s);
    }
}

std::vector<TaskSample> Metrics::collect() const {
    std::vector<TaskSample> all;

    std::size_t total = 0;
    for (const auto& shard : worker_shards_) total += shard->samples.size();
    {
        std::lock_guard<std::mutex> lock(external_mutex_);
        total += external_shard_.samples.size();
    }
    all.reserve(total);

    for (const auto& shard : worker_shards_) {
        all.insert(all.end(), shard->samples.begin(), shard->samples.end());
    }
    {
        std::lock_guard<std::mutex> lock(external_mutex_);
        all.insert(all.end(), external_shard_.samples.begin(), external_shard_.samples.end());
    }
    return all;
}

LatencySummary summarize_samples(const std::vector<TaskSample>& samples) {
    std::vector<uint64_t> queue_wait, exec, total;
    queue_wait.reserve(samples.size());
    exec.reserve(samples.size());
    total.reserve(samples.size());

    for (const TaskSample& s : samples) {
        queue_wait.push_back(s.queue_wait_ns());
        exec.push_back(s.exec_ns());
        total.push_back(s.total_ns());
    }

    LatencySummary out;
    out.queue_wait = compute_stats(std::move(queue_wait));
    out.exec       = compute_stats(std::move(exec));
    out.total      = compute_stats(std::move(total));
    return out;
}

LatencySummary Metrics::summarize() const {
    return summarize_samples(collect());
}

static void print_stats_line(const char* label, const LatencyStats& st) {
    // Nanoseconds are the storage unit; microseconds read better in a terminal.
    std::printf("    %-11s min=%.1f  p50=%.1f  p95=%.1f  p99=%.1f  max=%.1f  mean=%.1f (us)\n",
                label,
                st.min / 1000.0, st.p50 / 1000.0, st.p95 / 1000.0,
                st.p99 / 1000.0, st.max / 1000.0, st.mean / 1000.0);
}

void Metrics::print_summary() const {
    std::printf("  [Metrics] submitted=%" PRIu64 "  completed=%" PRIu64 "\n",
                submitted(), completed());

    std::vector<TaskSample> samples = collect();
    LatencySummary          sum     = summarize_samples(samples);
    if (sum.total.count == 0) {
        std::printf("    (no latency samples — tracing disabled)\n");
        return;
    }

    // How much work actually moved between workers. With no recursive spawning the
    // local Chase-Lev queues stay empty and all rebalancing happens via peer injection
    // queues, so the steal counters read zero while real work movement still shows up
    // here.
    uint64_t stolen = 0;
    for (const TaskSample& s : samples) {
        if (s.stolen) ++stolen;
    }
    std::printf("    samples=%" PRIu64 "  moved_between_workers=%" PRIu64 " (%.1f%%)\n",
                sum.total.count, stolen,
                sum.total.count > 0 ? (stolen * 100.0 / sum.total.count) : 0.0);

    // Per-worker task counts — the direct read on load balance.
    std::vector<uint64_t> per_worker(worker_shards_.size(), 0);
    uint64_t              external = 0;
    for (const TaskSample& s : samples) {
        if (s.worker_id >= 0 && s.worker_id < static_cast<int32_t>(per_worker.size())) {
            ++per_worker[static_cast<std::size_t>(s.worker_id)];
        } else {
            ++external;
        }
    }
    std::printf("    per-worker:");
    for (std::size_t i = 0; i < per_worker.size(); ++i) {
        std::printf(" w%zu=%" PRIu64, i, per_worker[i]);
    }
    if (external > 0) std::printf(" ext=%" PRIu64, external);
    std::printf("\n");

    print_stats_line("queue_wait", sum.queue_wait);
    print_stats_line("exec",       sum.exec);
    print_stats_line("total",      sum.total);
}

void Metrics::dump_csv(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "task_id,worker_id,stolen,submit_ns,start_ns,end_ns,"
                    "queue_wait_ns,exec_ns,total_ns\n");
    for (const TaskSample& s : collect()) {
        std::fprintf(f,
            "%" PRIu64 ",%" PRId32 ",%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            s.task_id, s.worker_id, s.stolen ? 1 : 0,
            s.submit_ns, s.start_ns, s.end_ns,
            s.queue_wait_ns(), s.exec_ns(), s.total_ns());
    }
    std::fclose(f);
}
