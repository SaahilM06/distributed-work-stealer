#include "net/Protocol.hpp"

namespace proto {

std::vector<uint8_t> frame(MsgType type, const std::vector<uint8_t>& payload) {
    ByteWriter w;
    w.u32(kMagic);
    w.u16(static_cast<uint16_t>(type));
    w.u16(0); // reserved
    w.u32(static_cast<uint32_t>(payload.size()));
    w.bytes(payload.data(), payload.size());
    return w.buf();
}

bool parse_header(const uint8_t* data, std::size_t size, MsgType& out_type, uint32_t& out_len) {
    if (size < kHeaderSize) return false;
    ByteReader r(data, size);
    if (r.u32() != kMagic) return false;
    uint16_t type = r.u16();
    r.u16(); // reserved
    uint32_t len = r.u32();
    if (!r.ok() || len > kMaxPayload) return false;
    out_type = static_cast<MsgType>(type);
    out_len  = len;
    return true;
}

void encode_task(ByteWriter& w, const Task& t) {
    w.u64(t.task_id);
    w.u64(t.parent_id);
    w.u32(t.depth);
    w.u32(t.cost_hint);
    w.u32(t.origin_node);
    w.u16(static_cast<uint16_t>(t.type));
    w.blob(t.payload);
}

Task decode_task(ByteReader& r) {
    Task t;
    t.task_id     = r.u64();
    t.parent_id   = r.u64();
    t.depth       = r.u32();
    t.cost_hint   = r.u32();
    t.origin_node = r.u32();
    t.type        = static_cast<TaskType>(r.u16());
    t.payload     = r.blob();
    return t;
}

static void encode_node_info(ByteWriter& w, const NodeInfo& n) {
    w.u32(n.node_id);
    w.str(n.host);
    w.u16(n.port);
    w.u32(n.num_workers);
    w.str(n.label);
    w.u64(n.pending);
}

static NodeInfo decode_node_info(ByteReader& r) {
    NodeInfo n;
    n.node_id     = r.u32();
    n.host        = r.str();
    n.port        = r.u16();
    n.num_workers = r.u32();
    n.label       = r.str();
    n.pending     = r.u64();
    return n;
}

std::vector<uint8_t> encode(const RegisterMsg& m) {
    ByteWriter w;
    w.str(m.host);
    w.u16(m.port);
    w.u32(m.num_workers);
    w.str(m.label);
    return w.buf();
}

std::vector<uint8_t> encode(const RegisterAckMsg& m) {
    ByteWriter w;
    w.u32(m.node_id);
    return w.buf();
}

std::vector<uint8_t> encode(const HeartbeatMsg& m) {
    ByteWriter w;
    w.u32(m.node_id);
    w.u64(m.pending);
    w.u64(m.completed);
    return w.buf();
}

std::vector<uint8_t> encode(const NodeListMsg& m) {
    ByteWriter w;
    w.u32(static_cast<uint32_t>(m.nodes.size()));
    for (const NodeInfo& n : m.nodes) encode_node_info(w, n);
    return w.buf();
}

std::vector<uint8_t> encode(const StealRequestMsg& m) {
    ByteWriter w;
    w.u32(m.requester_node);
    w.u32(m.max_tasks);
    w.u16(m.preferred_type);
    return w.buf();
}

std::vector<uint8_t> encode(const StealResponseMsg& m) {
    ByteWriter w;
    w.u32(static_cast<uint32_t>(m.tasks.size()));
    for (const Task& t : m.tasks) encode_task(w, t);
    w.u64(m.victim_pending);
    return w.buf();
}

std::vector<uint8_t> encode(const TaskResultMsg& m) {
    ByteWriter w;
    w.u64(m.task_id);
    w.u32(m.status);
    return w.buf();
}

bool decode(const std::vector<uint8_t>& b, RegisterMsg& m) {
    ByteReader r(b);
    m.host        = r.str();
    m.port        = r.u16();
    m.num_workers = r.u32();
    m.label       = r.str();
    return r.ok();
}

bool decode(const std::vector<uint8_t>& b, RegisterAckMsg& m) {
    ByteReader r(b);
    m.node_id = r.u32();
    return r.ok();
}

bool decode(const std::vector<uint8_t>& b, HeartbeatMsg& m) {
    ByteReader r(b);
    m.node_id   = r.u32();
    m.pending   = r.u64();
    m.completed = r.u64();
    return r.ok();
}

bool decode(const std::vector<uint8_t>& b, NodeListMsg& m) {
    ByteReader r(b);
    uint32_t count = r.u32();
    if (!r.ok()) return false;
    m.nodes.clear();
    for (uint32_t i = 0; i < count; ++i) {
        NodeInfo n = decode_node_info(r);
        if (!r.ok()) return false;
        m.nodes.push_back(std::move(n));
    }
    return r.ok();
}

bool decode(const std::vector<uint8_t>& b, StealRequestMsg& m) {
    ByteReader r(b);
    m.requester_node = r.u32();
    m.max_tasks      = r.u32();
    m.preferred_type = r.u16();
    return r.ok();
}

bool decode(const std::vector<uint8_t>& b, StealResponseMsg& m) {
    ByteReader r(b);
    uint32_t count = r.u32();
    if (!r.ok()) return false;
    m.tasks.clear();
    for (uint32_t i = 0; i < count; ++i) {
        Task t = decode_task(r);
        if (!r.ok()) return false;
        m.tasks.push_back(std::move(t));
    }
    m.victim_pending = r.u64();
    return r.ok();
}

bool decode(const std::vector<uint8_t>& b, TaskResultMsg& m) {
    ByteReader r(b);
    m.task_id = r.u64();
    m.status  = r.u32();
    return r.ok();
}

const char* msg_type_name(MsgType t) {
    switch (t) {
        case MsgType::Register:      return "Register";
        case MsgType::RegisterAck:   return "RegisterAck";
        case MsgType::Heartbeat:     return "Heartbeat";
        case MsgType::NodeList:      return "NodeList";
        case MsgType::StealRequest:  return "StealRequest";
        case MsgType::StealResponse: return "StealResponse";
        case MsgType::TaskResult:    return "TaskResult";
        case MsgType::Shutdown:      return "Shutdown";
    }
    return "Unknown";
}

} // namespace proto
