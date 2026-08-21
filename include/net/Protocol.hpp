#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "net/Serialize.hpp"
#include "runtime/Task.hpp"

namespace proto {

// TCP is a byte stream with no message boundaries, so every message is framed with a
// fixed header giving its type and length. Without this, two messages sent back to back
// arrive as one undifferentiated run of bytes.
constexpr uint32_t kMagic       = 0x48594452; // "HYDR"
constexpr std::size_t kHeaderSize = 12;       // magic(4) + type(2) + pad(2) + length(4)
constexpr uint32_t kMaxPayload  = 64u * 1024 * 1024;

enum class MsgType : uint16_t {
    Register = 1,   // node  -> coordinator
    RegisterAck,    // coordinator -> node
    Heartbeat,      // node  -> coordinator
    NodeList,       // coordinator -> node (reply to heartbeat)
    StealRequest,   // node  -> peer node
    StealResponse,  // peer node -> node
    TaskResult,     // thief -> origin node
    Shutdown,       // coordinator -> node
};

struct NodeInfo {
    uint32_t    node_id = 0;
    std::string host;
    uint16_t    port    = 0;
    uint32_t    num_workers = 0;
    std::string label;          // e.g. "gpu" / "cpu" — used by the adaptive scheduler
};

struct RegisterMsg {
    std::string host;
    uint16_t    port        = 0;
    uint32_t    num_workers = 0;
    std::string label;
};

struct RegisterAckMsg {
    uint32_t node_id = 0;
};

struct HeartbeatMsg {
    uint32_t node_id   = 0;
    uint64_t pending   = 0;   // queued + in-flight tasks; the load signal for stealing
    uint64_t completed = 0;
};

struct NodeListMsg {
    std::vector<NodeInfo> nodes;
};

struct StealRequestMsg {
    uint32_t requester_node = 0;
    uint32_t max_tasks      = 1;
};

struct StealResponseMsg {
    std::vector<Task> tasks;  // portable tasks only
};

struct TaskResultMsg {
    uint64_t task_id = 0;
    uint32_t status  = 0;     // 0 = ok
};

// ── framing ─────────────────────────────────────────────────────────────────
std::vector<uint8_t> frame(MsgType type, const std::vector<uint8_t>& payload);
bool parse_header(const uint8_t* data, std::size_t size, MsgType& out_type, uint32_t& out_len);

// ── encode / decode ─────────────────────────────────────────────────────────
// Only a task's portable form crosses the wire; `fn` is deliberately not encoded
// because a closure cannot be meaningfully reconstructed in another process.
void encode_task(ByteWriter& w, const Task& t);
Task decode_task(ByteReader& r);

std::vector<uint8_t> encode(const RegisterMsg& m);
std::vector<uint8_t> encode(const RegisterAckMsg& m);
std::vector<uint8_t> encode(const HeartbeatMsg& m);
std::vector<uint8_t> encode(const NodeListMsg& m);
std::vector<uint8_t> encode(const StealRequestMsg& m);
std::vector<uint8_t> encode(const StealResponseMsg& m);
std::vector<uint8_t> encode(const TaskResultMsg& m);

bool decode(const std::vector<uint8_t>& b, RegisterMsg& m);
bool decode(const std::vector<uint8_t>& b, RegisterAckMsg& m);
bool decode(const std::vector<uint8_t>& b, HeartbeatMsg& m);
bool decode(const std::vector<uint8_t>& b, NodeListMsg& m);
bool decode(const std::vector<uint8_t>& b, StealRequestMsg& m);
bool decode(const std::vector<uint8_t>& b, StealResponseMsg& m);
bool decode(const std::vector<uint8_t>& b, TaskResultMsg& m);

const char* msg_type_name(MsgType t);

} // namespace proto
