#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// One row per executed task. Timestamps are nanoseconds since the owning Metrics
// object was constructed, so they stay small and are directly comparable within a run.
struct TaskSample {
    uint64_t task_id   = 0;
    uint64_t submit_ns = 0;  // when submit() stamped it
    uint64_t start_ns  = 0;  // when a worker began executing it
    uint64_t end_ns    = 0;  // when it finished
    int32_t  worker_id = -1; // -1 = run by a non-worker thread (a thread helping out
                             //      from inside Future::get())
    bool     stolen    = false;

    // Time spent waiting in a queue — the signal that actually reflects scheduling
    // quality. Rising p95/p99 queue wait under load is what work stealing is supposed
    // to prevent, so this is the number the later adaptive-scheduler phases are judged on.
    uint64_t queue_wait_ns() const { return start_ns - submit_ns; }
    uint64_t exec_ns()       const { return end_ns   - start_ns;  }
    uint64_t total_ns()      const { return end_ns   - submit_ns; }
};

struct LatencyStats {
    uint64_t count = 0;
    uint64_t min   = 0;
    uint64_t p50   = 0;
    uint64_t p95   = 0;
    uint64_t p99   = 0;
    uint64_t max   = 0;
    double   mean  = 0.0;
};

struct LatencySummary {
    LatencyStats queue_wait;
    LatencyStats exec;
    LatencyStats total;
};

// `values` is taken by value and sorted in place. Percentiles use nearest-rank.
LatencyStats compute_stats(std::vector<uint64_t> values);

LatencySummary summarize_samples(const std::vector<TaskSample>& samples);

class Metrics {
public:
    // Per-executing-thread sample buffer. Each worker appends only to its own shard,
    // so recording never contends — mirroring the per-worker deque design.
    struct Shard {
        int                     worker_id = -1;
        std::vector<TaskSample> samples;
    };

    explicit Metrics(int num_worker_shards);

    void     record_submitted();
    void     record_completed();
    uint64_t submitted() const;
    uint64_t completed() const;

    // Nanoseconds since this Metrics was constructed.
    uint64_t now_ns() const;

    // Per-task sampling costs two clock reads plus an append per task, which is
    // measurable at multi-million-task/sec throughput — so it can be turned off to
    // benchmark the scheduler itself. The submitted/completed counters are always on.
    void enable_tracing(bool on) { tracing_.store(on, std::memory_order_relaxed); }
    bool tracing_enabled() const { return tracing_.load(std::memory_order_relaxed); }

    // Called by each worker thread at start/exit so record_task() can find its shard.
    void bind_worker_shard(int worker_id);
    void unbind_worker_shard();

    void record_task(uint64_t task_id, uint64_t submit_ns, uint64_t start_ns,
                     uint64_t end_ns, bool stolen);

    // Merge every shard. Only safe once all workers have joined — worker shards are
    // appended to without locking, which is the whole point of sharding them.
    std::vector<TaskSample> collect() const;

    LatencySummary summarize() const;

    void print_summary() const;
    void dump_csv(const std::string& path) const;

private:
    // Bounds memory on long runs; samples past this per shard are dropped rather than
    // growing without limit.
    static constexpr std::size_t kMaxSamplesPerShard = 1000000;

    static thread_local Shard* t_shard_;

    std::vector<std::unique_ptr<Shard>> worker_shards_;
    Shard                               external_shard_;
    mutable std::mutex                  external_mutex_;

    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<bool>     tracing_{true};

    std::chrono::steady_clock::time_point origin_;
};
