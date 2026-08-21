#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference/InferenceJob.hpp"
#include "metrics/Metrics.hpp"
#include "runtime/Runtime.hpp"

namespace inference {

enum class JobState : uint8_t { Queued = 0, Running, Done };

struct JobStatus {
    uint64_t  request_id = 0;
    JobState  state      = JobState::Queued;
    Stage     stage      = Stage::Decode;
    ModelKind kind       = ModelKind::ImageClassifier;
    uint64_t  submit_ns  = 0;
    uint64_t  finish_ns  = 0;
    uint64_t  latency_ns() const { return finish_ns > submit_ns ? finish_ns - submit_ns : 0; }
};

// Turns a client request into the four-stage task chain and tracks it to completion.
//
// The stages form a dependency chain, not a fork-join: postprocess cannot start until
// inference is done. Rather than blocking a worker on a Future between stages, each
// stage's task submits the next stage when it finishes. A request in flight therefore
// occupies no thread at all while it waits — only while it is actually computing,
// which is what lets a handful of workers carry far more concurrent requests than
// there are threads.
class JobManager {
public:
    JobManager(Runtime& rt, std::string node_label);

    // Registers the stage handlers. Must be called on every node in the cluster, and
    // before the runtime starts taking work.
    static void register_handlers(const std::string& node_label);

    uint64_t submit(const Request& req);

    bool status(uint64_t request_id, JobStatus& out) const;

    // Blocks until every submitted request has finished.
    void wait_all() { rt_.wait_all(); }

    uint64_t completed() const { return completed_.load(std::memory_order_relaxed); }
    uint64_t submitted() const { return submitted_.load(std::memory_order_relaxed); }

    // End-to-end latency per request, in nanoseconds — the number that actually
    // matters to a client, as distinct from how long any individual stage took.
    std::vector<uint64_t> request_latencies_ns() const;

    // Per-request rows for external analysis.
    void dump_csv(const std::string& path) const;

private:
    // Submits `stage` for a request; the handler submits the following stage when it
    // completes, walking the chain to Postprocess.
    void submit_stage(const StagePayload& p);
    void on_stage_done(const StagePayload& p);

    Runtime&    rt_;
    std::string node_label_;

    mutable std::mutex                            mutex_;
    std::unordered_map<uint64_t, JobStatus>       jobs_;
    std::atomic<uint64_t> next_request_id_{1};
    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};
};

} // namespace inference
