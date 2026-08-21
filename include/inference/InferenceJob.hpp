#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "net/Serialize.hpp"
#include "runtime/Task.hpp"

namespace inference {

// What kind of model a request targets. Cost characteristics differ sharply between
// them, which is the whole reason the scheduler has anything interesting to do.
enum class ModelKind : uint8_t {
    ImageClassifier = 0,   // cost scales with pixels
    TextTransformer,       // cost scales super-linearly with sequence length
};

// The four stages of a request. Each becomes its own task, so stages of different
// requests can run on different nodes concurrently.
enum class Stage : uint8_t {
    Decode = 0,
    Preprocess,
    Infer,
    Postprocess,
};

const char* model_kind_name(ModelKind k);
const char* stage_name(Stage s);

// One client request. `image_pixels` or `sequence_length` is used depending on kind.
struct Request {
    uint64_t  request_id     = 0;
    ModelKind kind           = ModelKind::ImageClassifier;
    uint32_t  image_pixels   = 224 * 224;
    uint32_t  sequence_length = 128;
    uint32_t  batch_size     = 1;
};

// The work unit actually handed to the runtime: one stage of one request.
struct StagePayload {
    uint64_t  request_id     = 0;
    ModelKind kind           = ModelKind::ImageClassifier;
    Stage     stage          = Stage::Decode;
    uint32_t  image_pixels   = 0;
    uint32_t  sequence_length = 0;
    uint32_t  batch_size     = 1;
};

std::vector<uint8_t> encode_stage(const StagePayload& p);
bool                 decode_stage(const std::vector<uint8_t>& bytes, StagePayload& out);

// How expensive a stage is, in arbitrary "work units", derived from the request's
// shape rather than a flat constant. This is the cost model the scheduler is being
// tested against: it is deliberately non-uniform and non-obvious, so a static
// round-robin split cannot balance it in advance.
uint32_t stage_cost_units(const StagePayload& p);

// Maps a stage onto the TaskType used for capability routing. Inference is the stage a
// GPU-class node is dramatically better at; the rest are ordinary CPU work.
TaskType task_type_for(Stage s);

} // namespace inference
