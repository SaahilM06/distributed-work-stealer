#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inference {

// Runs a real ONNX model on the inference stage.
//
// Compiled only when HYDRA_WITH_ONNX is defined; everything above this class works
// unchanged without it, falling back to the simulated cost model. That keeps the
// project buildable on a machine with no ONNX Runtime while letting the same
// scheduler drive genuine model execution when it is available.
//
// Thread safety: an Ort::Session is safe to Run() concurrently from many threads, so a
// single instance is shared by every worker. That matters here — one session per
// worker would multiply the model's memory by the worker count for no benefit.
class OnnxModel {
public:
    OnnxModel();
    ~OnnxModel();

    OnnxModel(const OnnxModel&)            = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;

    // Returns false (with a reason on stderr) if the model cannot be loaded, so the
    // caller can fall back to simulated cost rather than failing outright.
    bool load(const std::string& model_path, int intra_op_threads = 1);

    bool loaded() const;

    // Runs one forward pass on a synthetic input of `batch` images. The input values
    // are arbitrary: this measures scheduling of real model execution, and the
    // arithmetic cost of a convolution does not depend on the pixel values.
    // Returns false if the run failed.
    bool run(uint32_t batch);

    const std::string& path() const;
    uint64_t           runs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Process-wide model used by the inference stage handler. Loaded once at startup,
// then read concurrently by every worker.
OnnxModel& global_model();

} // namespace inference
