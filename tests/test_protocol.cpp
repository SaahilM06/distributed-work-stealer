#include "net/Protocol.hpp"
#include "net/Socket.hpp"
#include "runtime/TaskRegistry.hpp"

#include <atomic>
#include "Check.hpp"
#include <cstdio>
#include <thread>

using namespace proto;

static void test_byte_roundtrip() {
    ByteWriter w;
    w.u8(0xAB);
    w.u16(0x1234);
    w.u32(0xDEADBEEF);
    w.u64(0x0123456789ABCDEFull);
    w.str("hydra");
    w.blob({1, 2, 3, 4});

    ByteReader r(w.buf());
    CHECK(r.u8() == 0xAB);
    CHECK(r.u16() == 0x1234);
    CHECK(r.u32() == 0xDEADBEEF);
    CHECK(r.u64() == 0x0123456789ABCDEFull);
    CHECK(r.str() == "hydra");
    CHECK((r.blob() == std::vector<uint8_t>{1, 2, 3, 4}));
    CHECK(r.ok());
    CHECK(r.remaining() == 0);
    std::printf("PASS test_byte_roundtrip\n");
}

// A truncated buffer must fail cleanly rather than read past the end — this parses
// bytes that arrived from the network.
static void test_truncated_input_is_rejected() {
    ByteWriter w;
    w.u64(42);
    w.str("some-fairly-long-string");

    std::vector<uint8_t> truncated = w.buf();
    truncated.resize(truncated.size() / 2);

    ByteReader r(truncated);
    r.u64();
    r.str();
    CHECK(!r.ok());

    // A length prefix larger than the buffer must not be trusted.
    ByteWriter bad;
    bad.u32(0xFFFFFF);   // claims 16MB of string
    bad.u8(1);
    ByteReader r2(bad.buf());
    CHECK(r2.str().empty());
    CHECK(!r2.ok());
    std::printf("PASS test_truncated_input_is_rejected\n");
}

static void test_framing() {
    std::vector<uint8_t> payload{9, 8, 7};
    std::vector<uint8_t> framed = frame(MsgType::Heartbeat, payload);
    CHECK(framed.size() == kHeaderSize + payload.size());

    MsgType  type = MsgType::Register;
    uint32_t len  = 0;
    CHECK(parse_header(framed.data(), framed.size(), type, len));
    CHECK(type == MsgType::Heartbeat);
    CHECK(len == payload.size());

    // Garbage must be rejected via the magic number.
    std::vector<uint8_t> garbage(kHeaderSize, 0x00);
    CHECK(!parse_header(garbage.data(), garbage.size(), type, len));
    // A short read must not be parsed as a header.
    CHECK(!parse_header(framed.data(), 3, type, len));
    std::printf("PASS test_framing\n");
}

static void test_message_roundtrip() {
    {
        RegisterMsg in{"127.0.0.1", 9001, 8, "gpu"};
        RegisterMsg out;
        CHECK(decode(encode(in), out));
        CHECK(out.host == "127.0.0.1" && out.port == 9001);
        CHECK(out.num_workers == 8 && out.label == "gpu");
    }
    {
        HeartbeatMsg in{7, 123, 456};
        HeartbeatMsg out;
        CHECK(decode(encode(in), out));
        CHECK(out.node_id == 7 && out.pending == 123 && out.completed == 456);
    }
    {
        NodeListMsg in;
        in.nodes.push_back({1, "10.0.0.1", 9001, 4, "cpu"});
        in.nodes.push_back({2, "10.0.0.2", 9002, 8, "gpu"});
        NodeListMsg out;
        CHECK(decode(encode(in), out));
        CHECK(out.nodes.size() == 2);
        CHECK(out.nodes[1].node_id == 2 && out.nodes[1].label == "gpu");
        CHECK(out.nodes[1].port == 9002 && out.nodes[1].num_workers == 8);
    }
    {
        TaskResultMsg in{99, 0};
        TaskResultMsg out;
        CHECK(decode(encode(in), out));
        CHECK(out.task_id == 99 && out.status == 0);
    }
    std::printf("PASS test_message_roundtrip\n");
}

static void test_task_roundtrip() {
    StealResponseMsg in;
    Task t;
    t.task_id     = 77;
    t.parent_id   = 12;
    t.depth       = 3;
    t.cost_hint   = 250;
    t.origin_node = 5;
    t.type        = TaskType::SyntheticCompute;
    t.payload     = {0xDE, 0xAD, 0xBE, 0xEF};
    in.tasks.push_back(t);

    StealResponseMsg out;
    CHECK(decode(encode(in), out));
    CHECK(out.tasks.size() == 1);

    const Task& d = out.tasks[0];
    CHECK(d.task_id == 77 && d.parent_id == 12 && d.depth == 3);
    CHECK(d.cost_hint == 250 && d.origin_node == 5);
    CHECK(d.type == TaskType::SyntheticCompute);
    CHECK((d.payload == std::vector<uint8_t>{0xDE, 0xAD, 0xBE, 0xEF}));
    CHECK(d.portable());   // no fn, so it may cross the wire
    std::printf("PASS test_task_roundtrip\n");
}

static void test_registry() {
    TaskRegistry& reg = TaskRegistry::instance();
    reg.clear();

    std::atomic<int> sum{0};
    reg.register_handler(TaskType::SyntheticCompute, [&sum](const std::vector<uint8_t>& p) {
        for (uint8_t b : p) sum.fetch_add(b, std::memory_order_relaxed);
    });

    CHECK(reg.has_handler(TaskType::SyntheticCompute));
    CHECK(!reg.has_handler(TaskType::MandelbrotTile));

    Task t;
    t.type    = TaskType::SyntheticCompute;
    t.payload = {10, 20, 30};
    CHECK(run_task(t));
    CHECK(sum.load() == 60);

    // No handler registered → run_task reports failure rather than silently dropping
    // the task, which on a real node would mean the cluster's tables disagree.
    Task missing;
    missing.type = TaskType::MandelbrotTile;
    CHECK(!run_task(missing));

    // A closure task still runs through the same entry point.
    bool ran = false;
    Task local;
    local.fn = [&ran]() { ran = true; };
    CHECK(run_task(local));
    CHECK(ran);
    CHECK(!local.portable());   // has fn → local only

    reg.clear();
    std::printf("PASS test_registry\n");
}

// Real sockets over loopback: framing must survive an actual TCP stream, including
// two messages sent back to back (the case that breaks without length prefixing).
static void test_socket_message_exchange() {
    TcpSocket server;
    CHECK(server.listen(0));            // port 0 → kernel picks a free port
    uint16_t port = server.local_port();
    CHECK(port != 0);

    std::atomic<bool> server_ok{false};
    std::thread server_thread([&server, &server_ok]() {
        TcpSocket conn = server.accept();
        if (!conn.valid()) return;

        MsgType              type;
        std::vector<uint8_t> payload;

        if (!conn.recv_msg(type, payload) || type != MsgType::StealRequest) return;
        StealRequestMsg req;
        if (!decode(payload, req) || req.requester_node != 3) return;

        StealResponseMsg resp;
        Task t;
        t.task_id = 1234;
        t.type    = TaskType::SyntheticCompute;
        t.payload = {1, 2, 3};
        resp.tasks.push_back(t);

        if (!conn.send_msg(MsgType::StealResponse, encode(resp))) return;
        // Second message immediately after, no delay: arrives coalesced in the same
        // TCP segment, so only correct framing separates them.
        if (!conn.send_msg(MsgType::TaskResult, encode(TaskResultMsg{1234, 0}))) return;
        server_ok.store(true);
    });

    TcpSocket client;
    CHECK(client.connect("127.0.0.1", port));
    CHECK(client.send_msg(MsgType::StealRequest, encode(StealRequestMsg{3, 1})));

    MsgType              type;
    std::vector<uint8_t> payload;

    CHECK(client.recv_msg(type, payload));
    CHECK(type == MsgType::StealResponse);
    StealResponseMsg resp;
    CHECK(decode(payload, resp));
    CHECK(resp.tasks.size() == 1 && resp.tasks[0].task_id == 1234);

    CHECK(client.recv_msg(type, payload));
    CHECK(type == MsgType::TaskResult);
    TaskResultMsg result;
    CHECK(decode(payload, result));
    CHECK(result.task_id == 1234);

    server_thread.join();
    CHECK(server_ok.load());
    std::printf("PASS test_socket_message_exchange\n");
}

int main() {
    test_byte_roundtrip();
    test_truncated_input_is_rejected();
    test_framing();
    test_message_roundtrip();
    test_task_roundtrip();
    test_registry();
    test_socket_message_exchange();
    std::printf("All tests passed.\n");
    return 0;
}
