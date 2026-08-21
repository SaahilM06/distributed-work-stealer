#include "inference/OnnxModel.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>

#ifdef HYDRA_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace inference {

#ifdef HYDRA_WITH_ONNX

struct OnnxModel::Impl {
    Ort::Env                       env{ORT_LOGGING_LEVEL_WARNING, "hydra"};
    std::unique_ptr<Ort::Session>  session;
    Ort::AllocatorWithDefaultOptions alloc;

    std::string              path;
    std::string              input_name;
    std::string              output_name;
    std::vector<int64_t>     input_shape;   // with batch as dim 0
    std::vector<float>       scratch;       // reused input buffer
    std::mutex               scratch_mutex;
    std::atomic<uint64_t>    runs{0};
};

OnnxModel::OnnxModel() : impl_(std::make_unique<Impl>()) {}
OnnxModel::~OnnxModel() = default;

bool OnnxModel::load(const std::string& model_path, int intra_op_threads) {
    try {
        Ort::SessionOptions opts;
        // One thread per session run. The runtime already has its own workers, and
        // letting ORT spawn its own pool on top would oversubscribe the machine and
        // make the scheduling measurements meaningless — parallelism here comes from
        // running many requests at once, not from splitting one.
        opts.SetIntraOpNumThreads(intra_op_threads);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, model_path.c_str(), opts);

        Ort::AllocatedStringPtr in  = impl_->session->GetInputNameAllocated(0, impl_->alloc);
        Ort::AllocatedStringPtr out = impl_->session->GetOutputNameAllocated(0, impl_->alloc);
        impl_->input_name  = in.get();
        impl_->output_name = out.get();

        Ort::TypeInfo info = impl_->session->GetInputTypeInfo(0);
        impl_->input_shape = info.GetTensorTypeAndShapeInfo().GetShape();

        // A dynamic axis comes back as -1; pin everything except batch (dim 0), which
        // the caller varies per request.
        for (std::size_t i = 1; i < impl_->input_shape.size(); ++i) {
            if (impl_->input_shape[i] < 0) impl_->input_shape[i] = 224;
        }
        if (!impl_->input_shape.empty()) impl_->input_shape[0] = 1;

        impl_->path = model_path;

        std::printf("[onnx] loaded %s  input=%s", model_path.c_str(), impl_->input_name.c_str());
        for (int64_t d : impl_->input_shape) std::printf(" %lld", (long long)d);
        std::printf("  output=%s\n", impl_->output_name.c_str());
        std::fflush(stdout);
        return true;
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[onnx] failed to load %s: %s\n", model_path.c_str(), e.what());
        impl_->session.reset();
        return false;
    }
}

bool OnnxModel::loaded() const { return impl_->session != nullptr; }

bool OnnxModel::run(uint32_t batch) {
    if (!impl_->session) return false;
    if (batch == 0) batch = 1;

    try {
        std::vector<int64_t> shape = impl_->input_shape;
        if (shape.empty()) return false;
        shape[0] = static_cast<int64_t>(batch);

        std::size_t elems = 1;
        for (int64_t d : shape) elems *= static_cast<std::size_t>(d);

        // Every worker runs its own forward pass, so the input buffer must be
        // per-call — sharing one across threads would corrupt concurrent runs.
        std::vector<float> input(elems, 0.5f);

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), shape.data(), shape.size());

        const char* in_names[]  = {impl_->input_name.c_str()};
        const char* out_names[] = {impl_->output_name.c_str()};

        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

        impl_->runs.fetch_add(1, std::memory_order_relaxed);
        return !outputs.empty();
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[onnx] run failed: %s\n", e.what());
        return false;
    }
}

const std::string& OnnxModel::path() const { return impl_->path; }
uint64_t OnnxModel::runs() const { return impl_->runs.load(std::memory_order_relaxed); }

#else  // ── built without ONNX Runtime ──────────────────────────────────────

struct OnnxModel::Impl {
    std::string path;
};

OnnxModel::OnnxModel() : impl_(std::make_unique<Impl>()) {}
OnnxModel::~OnnxModel() = default;

bool OnnxModel::load(const std::string& model_path, int) {
    std::fprintf(stderr,
                 "[onnx] built without ONNX Runtime; ignoring --model %s "
                 "(reconfigure with -DHYDRA_WITH_ONNX=ON)\n",
                 model_path.c_str());
    return false;
}

bool OnnxModel::loaded() const { return false; }
bool OnnxModel::run(uint32_t) { return false; }
const std::string& OnnxModel::path() const { return impl_->path; }
uint64_t OnnxModel::runs() const { return 0; }

#endif

OnnxModel& global_model() {
    static OnnxModel model;
    return model;
}

} // namespace inference
