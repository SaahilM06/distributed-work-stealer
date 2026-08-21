#include "inference/InferenceJob.hpp"

namespace inference {

const char* model_kind_name(ModelKind k) {
    switch (k) {
        case ModelKind::ImageClassifier: return "image";
        case ModelKind::TextTransformer: return "text";
    }
    return "unknown";
}

const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Decode:      return "decode";
        case Stage::Preprocess:  return "preprocess";
        case Stage::Infer:       return "infer";
        case Stage::Postprocess: return "postprocess";
    }
    return "unknown";
}

std::vector<uint8_t> encode_stage(const StagePayload& p) {
    ByteWriter w;
    w.u64(p.request_id);
    w.u8(static_cast<uint8_t>(p.kind));
    w.u8(static_cast<uint8_t>(p.stage));
    w.u32(p.image_pixels);
    w.u32(p.sequence_length);
    w.u32(p.batch_size);
    return w.buf();
}

bool decode_stage(const std::vector<uint8_t>& bytes, StagePayload& out) {
    ByteReader r(bytes);
    out.request_id      = r.u64();
    out.kind            = static_cast<ModelKind>(r.u8());
    out.stage           = static_cast<Stage>(r.u8());
    out.image_pixels    = r.u32();
    out.sequence_length = r.u32();
    out.batch_size      = r.u32();
    return r.ok();
}

uint32_t stage_cost_units(const StagePayload& p) {
    // Numbers chosen to mirror the shape of real inference cost, not to match any
    // specific model:
    //   - decode/postprocess are small and roughly fixed
    //   - preprocess scales linearly with input size (resize, normalise, tokenise)
    //   - inference dominates, and for a transformer grows with the square of the
    //     sequence length because attention is quadratic
    // The quadratic term is the important part: it means two requests that look
    // similar can differ in cost by orders of magnitude, which is precisely what
    // defeats splitting work up front.
    uint64_t units = 0;

    switch (p.kind) {
        case ModelKind::ImageClassifier: {
            uint64_t px = p.image_pixels ? p.image_pixels : 1;
            switch (p.stage) {
                case Stage::Decode:      units = 200 + px / 64;  break;
                case Stage::Preprocess:  units = 300 + px / 16;  break;
                case Stage::Infer:       units = 2000 + px / 2;  break;
                case Stage::Postprocess: units = 150;            break;
            }
            break;
        }
        case ModelKind::TextTransformer: {
            uint64_t len = p.sequence_length ? p.sequence_length : 1;
            switch (p.stage) {
                case Stage::Decode:      units = 100 + len;                 break;
                case Stage::Preprocess:  units = 200 + len * 4;             break;
                // Quadratic in sequence length — the attention cost.
                case Stage::Infer:       units = 1000 + (len * len) / 4;    break;
                case Stage::Postprocess: units = 120 + len / 2;             break;
            }
            break;
        }
    }

    units *= (p.batch_size ? p.batch_size : 1);
    if (units > 0xFFFFFFFFull) units = 0xFFFFFFFFull;
    return static_cast<uint32_t>(units);
}

TaskType task_type_for(Stage s) {
    // MandelbrotTile is reused as the "inference" tag so that gpu-labelled nodes,
    // which already advertise a preference for it, attract the inference stage.
    return (s == Stage::Infer) ? TaskType::MandelbrotTile : TaskType::SyntheticCompute;
}

} // namespace inference
