#include "inference/JobManager.hpp"
#include "inference/OnnxModel.hpp"
#include "runtime/TaskRegistry.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace inference {

// Burns CPU proportional to the stage's cost. Used for the stages that stay simulated,
// and for the inference stage when no ONNX model is loaded.
static void burn(uint32_t units) {
    volatile uint64_t x = 0;
    for (uint32_t i = 0; i < units; ++i) x += i;
}

void JobManager::register_handlers(const std::string& node_label,
                                   const std::string& model_path) {
    // A "gpu" node runs the inference stage far faster than a cpu node. Everything
    // else runs at the same speed everywhere. This asymmetry is what makes capability
    // routing worth doing at all — without it, every node is interchangeable and the
    // adaptive policy has nothing to be smart about.
    const uint32_t infer_divisor = (node_label == "gpu") ? 8u : 1u;

    bool use_model = false;
    if (!model_path.empty()) {
        use_model = global_model().load(model_path, /*intra_op_threads=*/1);
        if (!use_model) {
            std::fprintf(stderr,
                         "[inference] falling back to the simulated cost model\n");
        }
    }

    auto run_stage = [infer_divisor, use_model](const std::vector<uint8_t>& payload) {
        StagePayload p;
        if (!decode_stage(payload, p)) return;

        // The inference stage is the one that becomes real: decode/preprocess/
        // postprocess remain simulated, since the interesting scheduling question is
        // about the expensive stage and where it runs.
        if (use_model && p.stage == Stage::Infer && p.kind == ModelKind::ImageClassifier) {
            // Batch is capped so one request cannot monopolise a worker for seconds —
            // a real serving system would enforce the same bound.
            uint32_t batch = p.batch_size ? p.batch_size : 1;
            if (batch > 8) batch = 8;
            if (global_model().run(batch)) return;
            // A failed run falls through to the simulated path rather than silently
            // completing a stage that did no work.
        }

        uint32_t units = stage_cost_units(p);
        if (p.stage == Stage::Infer) units /= infer_divisor;
        burn(units);
    };

    TaskRegistry::instance().register_handler(TaskType::SyntheticCompute, run_stage);
    TaskRegistry::instance().register_handler(TaskType::MandelbrotTile,   run_stage);
}

JobManager::JobManager(Runtime& rt, std::string node_label)
    : rt_(rt), node_label_(std::move(node_label))
{
    // Every finished task reports back here so the next stage can be submitted. This
    // fires whether the stage ran locally or was stolen and run on another node, which
    // is what keeps a request's chain intact across the cluster.
    rt_.set_task_observer([this](const Task& t) {
        if (t.payload.empty()) return;   // not one of ours (e.g. a settled re-run)
        StagePayload p;
        if (!decode_stage(t.payload, p)) return;
        on_stage_done(p);
    });
}

uint64_t JobManager::submit(const Request& req) {
    uint64_t id = next_request_id_.fetch_add(1, std::memory_order_relaxed);

    StagePayload p;
    p.request_id      = id;
    p.kind            = req.kind;
    p.stage           = Stage::Decode;
    p.image_pixels    = req.image_pixels;
    p.sequence_length = req.sequence_length;
    p.batch_size      = req.batch_size;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        JobStatus st;
        st.request_id = id;
        st.state      = JobState::Running;
        st.stage      = Stage::Decode;
        st.kind       = req.kind;
        st.submit_ns  = rt_.metrics().now_ns();
        jobs_.emplace(id, st);
    }

    submitted_.fetch_add(1, std::memory_order_relaxed);
    submit_stage(p);
    return id;
}

void JobManager::submit_stage(const StagePayload& p) {
    rt_.submit_portable(task_type_for(p.stage), encode_stage(p), stage_cost_units(p));
}

void JobManager::on_stage_done(const StagePayload& p) {
    // Advance the chain. Each stage submits its successor when it completes, so a
    // request in flight never occupies a thread while waiting — unlike blocking on a
    // Future between stages, which would pin one worker per concurrent request.
    switch (p.stage) {
        case Stage::Decode: {
            StagePayload next = p;
            next.stage = Stage::Preprocess;
            submit_stage(next);
            break;
        }
        case Stage::Preprocess: {
            StagePayload next = p;
            next.stage = Stage::Infer;
            submit_stage(next);
            break;
        }
        case Stage::Infer: {
            StagePayload next = p;
            next.stage = Stage::Postprocess;
            submit_stage(next);
            break;
        }
        case Stage::Postprocess: {
            // End of the chain: the request is finished.
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = jobs_.find(p.request_id);
            if (it != jobs_.end() && it->second.state != JobState::Done) {
                it->second.state     = JobState::Done;
                it->second.stage     = Stage::Postprocess;
                it->second.finish_ns = rt_.metrics().now_ns();
                completed_.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(p.request_id);
    if (it != jobs_.end()) {
        it->second.stage = static_cast<Stage>(static_cast<uint8_t>(p.stage) + 1);
    }
}

bool JobManager::status(uint64_t request_id, JobStatus& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(request_id);
    if (it == jobs_.end()) return false;
    out = it->second;
    return true;
}

std::vector<uint64_t> JobManager::request_latencies_ns() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64_t> out;
    out.reserve(jobs_.size());
    for (const auto& kv : jobs_) {
        if (kv.second.state == JobState::Done) out.push_back(kv.second.latency_ns());
    }
    return out;
}

void JobManager::dump_csv(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "request_id,model,state,submit_ns,finish_ns,latency_ns\n");

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& kv : jobs_) {
        const JobStatus& j = kv.second;
        std::fprintf(f, "%" PRIu64 ",%s,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                     j.request_id, model_kind_name(j.kind),
                     static_cast<int>(j.state), j.submit_ns, j.finish_ns, j.latency_ns());
    }
    std::fclose(f);
}

} // namespace inference
