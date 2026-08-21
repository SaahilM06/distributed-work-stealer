#include "Check.hpp"
#include "cluster/Coordinator.hpp"
#include "cluster/Node.hpp"
#include "inference/InferenceJob.hpp"
#include "inference/JobManager.hpp"
#include "inference/OnnxModel.hpp"
#include "runtime/Runtime.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace inference;

template <typename Pred>
static bool wait_until(Pred pred, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

static void test_stage_payload_roundtrip() {
    StagePayload p;
    p.request_id      = 4242;
    p.kind            = ModelKind::TextTransformer;
    p.stage           = Stage::Infer;
    p.image_pixels    = 512 * 512;
    p.sequence_length = 777;
    p.batch_size      = 3;

    StagePayload out;
    CHECK(decode_stage(encode_stage(p), out));
    CHECK(out.request_id == 4242);
    CHECK(out.kind == ModelKind::TextTransformer);
    CHECK(out.stage == Stage::Infer);
    CHECK(out.sequence_length == 777);
    CHECK(out.batch_size == 3);
    std::printf("PASS test_stage_payload_roundtrip\n");
}

// The cost model has to actually be uneven, otherwise there is nothing for a scheduler
// to get right and the whole benchmark is meaningless.
static void test_cost_model_is_skewed() {
    StagePayload small;
    small.kind = ModelKind::TextTransformer;
    small.stage = Stage::Infer;
    small.sequence_length = 64;

    StagePayload large = small;
    large.sequence_length = 1024;

    uint32_t c_small = stage_cost_units(small);
    uint32_t c_large = stage_cost_units(large);

    // 16x the sequence length should cost far more than 16x, because attention is
    // quadratic — that super-linearity is what makes request cost unpredictable.
    CHECK(c_large > c_small * 100);

    // Inference must dominate the cheap stages, or routing it by capability is pointless.
    StagePayload decode = large;
    decode.stage = Stage::Decode;
    CHECK(stage_cost_units(decode) < c_large / 10);

    std::printf("PASS test_cost_model_is_skewed: seq64=%u seq1024=%u units\n", c_small, c_large);
}

// Every request must walk all four stages exactly once and end up Done.
static void test_pipeline_runs_all_stages() {
    constexpr int NUM_REQUESTS = 40;

    JobManager::register_handlers("cpu");
    Runtime rt(4);
    JobManager jobs(rt, "cpu");

    std::vector<uint64_t> ids;
    for (int i = 0; i < NUM_REQUESTS; ++i) {
        Request r;
        r.kind            = (i % 2) ? ModelKind::TextTransformer : ModelKind::ImageClassifier;
        r.sequence_length = 128;
        r.image_pixels    = 224 * 224;
        ids.push_back(jobs.submit(r));
    }

    jobs.wait_all();
    rt.shutdown();

    CHECK(jobs.completed() == (uint64_t)NUM_REQUESTS);

    // Four stages per request: the chain really ran end to end rather than stopping
    // partway and being counted done.
    CHECK(rt.local_executed() == (uint64_t)NUM_REQUESTS * 4);

    for (uint64_t id : ids) {
        JobStatus st;
        CHECK(jobs.status(id, st));
        CHECK(st.state == JobState::Done);
        CHECK(st.stage == Stage::Postprocess);
        CHECK(st.latency_ns() > 0);
    }
    std::printf("PASS test_pipeline_runs_all_stages: %d requests x 4 stages\n", NUM_REQUESTS);
}

// A pipeline whose stages get stolen must still complete, and must still complete
// exactly once per stage — the chain is advanced by the origin even when another node
// did the work.
static void test_pipeline_survives_remote_execution() {
    // Sized so the batch represents tens of milliseconds of real work. The property
    // under test is that stolen stages are handled correctly; "a steal happened at all"
    // is only a precondition for that, and with a short batch it becomes a race against
    // the helper's steal-loop backoff rather than anything meaningful.
    constexpr int NUM_REQUESTS = 200;
    constexpr uint32_t SEQ_LEN = 1024;   // quadratic cost: ~260us per inference stage

    JobManager::register_handlers("cpu");

    Coordinator coord(0);
    CHECK(coord.start());

    NodeConfig helper_cfg;
    helper_cfg.coordinator_port = coord.port();
    helper_cfg.num_workers      = 2;
    helper_cfg.label            = "gpu";

    NodeConfig origin_cfg;
    origin_cfg.coordinator_port = coord.port();
    origin_cfg.num_workers      = 1;
    origin_cfg.label            = "cpu";
    origin_cfg.policy           = StealPolicy::None;

    Node helper(helper_cfg);
    Node origin(origin_cfg);
    CHECK(helper.start());
    CHECK(origin.start());
    helper.set_preferred_type(task_type_for(Stage::Infer));

    CHECK(wait_until([&]() { return helper.peer_count() >= 1 && origin.peer_count() >= 1; }, 5000));

    JobManager jobs(origin.runtime(), "cpu");

    for (int i = 0; i < NUM_REQUESTS; ++i) {
        Request r;
        r.kind            = ModelKind::TextTransformer;
        r.sequence_length = SEQ_LEN;
        jobs.submit(r);
    }
    jobs.wait_all();

    CHECK(jobs.completed() == (uint64_t)NUM_REQUESTS);
    CHECK(origin.tasks_stolen_out() > 0);   // stages really did leave the origin

    // Stages ran across both nodes but each exactly once.
    uint64_t total = origin.runtime().local_executed() + helper.runtime().local_executed();
    CHECK(total == (uint64_t)NUM_REQUESTS * 4);

    std::printf("PASS test_pipeline_survives_remote_execution: "
                "%llu stages here, %llu stolen away\n",
                (unsigned long long)origin.runtime().local_executed(),
                (unsigned long long)origin.tasks_stolen_out());

    origin.stop();
    helper.stop();
    coord.stop();
}

// Phase 12: the inference stage running a real ONNX model rather than a busy loop.
//
// Skipped, not failed, when the model file is absent or the build has no ONNX Runtime:
// the model is a 13MB artefact that is deliberately not in the repo, and every other
// guarantee in this suite holds regardless of whether it is present.
static void test_onnx_inference_stage() {
    const char* model_path = "models/mobilenet_v2.onnx";

    std::FILE* f = std::fopen(model_path, "rb");
    if (!f) {
        std::printf("SKIP test_onnx_inference_stage: no %s "
                    "(run: python3 scripts/export_model.py %s)\n", model_path, model_path);
        return;
    }
    std::fclose(f);

    JobManager::register_handlers("cpu", model_path);

    if (!global_model().loaded()) {
        std::printf("SKIP test_onnx_inference_stage: built without ONNX Runtime\n");
        return;
    }

    uint64_t runs_before = global_model().runs();

    constexpr int NUM_IMAGE = 6;
    constexpr int NUM_TEXT  = 4;

    Runtime rt(2);
    JobManager jobs(rt, "cpu");

    for (int i = 0; i < NUM_IMAGE; ++i) {
        Request r;
        r.kind         = ModelKind::ImageClassifier;
        r.image_pixels = 224 * 224;
        r.batch_size   = 1;
        jobs.submit(r);
    }
    for (int i = 0; i < NUM_TEXT; ++i) {
        Request r;
        r.kind            = ModelKind::TextTransformer;
        r.sequence_length = 128;
        jobs.submit(r);
    }

    jobs.wait_all();
    rt.shutdown();

    CHECK(jobs.completed() == (uint64_t)(NUM_IMAGE + NUM_TEXT));
    CHECK(rt.local_executed() == (uint64_t)(NUM_IMAGE + NUM_TEXT) * 4);

    // Exactly one forward pass per image request — the model ran for the inference
    // stage and only for it, not for decode/preprocess/postprocess.
    uint64_t ran = global_model().runs() - runs_before;
    CHECK(ran == (uint64_t)NUM_IMAGE);

    std::printf("PASS test_onnx_inference_stage: %llu real forward passes "
                "for %d image requests (%d text requests stayed simulated)\n",
                (unsigned long long)ran, NUM_IMAGE, NUM_TEXT);

    // Leave the registry on the simulated path so test ordering cannot matter.
    JobManager::register_handlers("cpu");
}

int main() {
    test_stage_payload_roundtrip();
    test_cost_model_is_skewed();
    test_pipeline_runs_all_stages();
    test_pipeline_survives_remote_execution();
    test_onnx_inference_stage();
    std::printf("All tests passed.\n");
    return 0;
}
