#include "cluster/Coordinator.hpp"
#include "cluster/Node.hpp"
#include "net/Serialize.hpp"
#include "runtime/TaskRegistry.hpp"

#include <atomic>
#include "Check.hpp"
#include <chrono>
#include <cstdio>
#include <thread>

// Runs a whole cluster inside one process: a coordinator plus two nodes on loopback.
// Not a substitute for the multi-process script, but it exercises registration, peer
// discovery, remote stealing, and completion accounting in something that can run in
// CI under a sanitizer.

static std::atomic<int> g_executed{0};

static std::vector<uint8_t> encode_cost(uint32_t iterations) {
    ByteWriter w;
    w.u32(iterations);
    return w.buf();
}

static void register_handlers() {
    TaskRegistry::instance().register_handler(
        TaskType::SyntheticCompute, [](const std::vector<uint8_t>& payload) {
            ByteReader r(payload);
            uint32_t iterations = r.u32();
            if (!r.ok()) return;
            volatile uint64_t x = 0;
            for (uint32_t i = 0; i < iterations; ++i) x += i;
            g_executed.fetch_add(1, std::memory_order_relaxed);
        });
}

// Polls until `pred` holds or the deadline passes; returns whether it held.
template <typename Pred>
static bool wait_until(Pred pred, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

static void test_cluster_remote_stealing() {
    constexpr int NUM_TASKS = 300;
    constexpr uint32_t COST = 20000;

    g_executed.store(0);

    Coordinator coord(0);          // port 0 → kernel assigns a free port
    CHECK(coord.start());
    uint16_t coord_port = coord.port();
    CHECK(coord_port != 0);

    // Thief: no work of its own, so anything it runs must have come over the network.
    NodeConfig thief_cfg;
    thief_cfg.coordinator_port = coord_port;
    thief_cfg.num_workers      = 2;
    thief_cfg.label            = "thief";

    // Origin: deliberately given a single worker so it cannot drain the pool alone.
    NodeConfig origin_cfg;
    origin_cfg.coordinator_port = coord_port;
    origin_cfg.num_workers      = 1;
    origin_cfg.label            = "origin";

    Node thief(thief_cfg);
    Node origin(origin_cfg);
    CHECK(thief.start());
    CHECK(origin.start());

    CHECK(thief.node_id() != 0 && origin.node_id() != 0);
    CHECK(thief.node_id() != origin.node_id());

    // Both nodes must learn about each other before stealing can happen.
    bool discovered = wait_until(
        [&]() { return thief.peer_count() >= 1 && origin.peer_count() >= 1; }, 5000);
    CHECK(discovered);

    for (int i = 0; i < NUM_TASKS; ++i) {
        origin.runtime().submit_portable(TaskType::SyntheticCompute, encode_cost(COST), COST);
    }
    origin.runtime().wait_all();

    uint64_t origin_ran = origin.runtime().local_executed();
    uint64_t thief_ran  = thief.runtime().local_executed();

    // wait_all() only returns once every task is accounted for, including the ones a
    // remote node ran and reported back — so the split must add up exactly.
    CHECK(origin_ran + thief_ran == (uint64_t)NUM_TASKS);
    CHECK(g_executed.load() == NUM_TASKS);

    // The whole point: work actually crossed the process boundary.
    CHECK(thief_ran > 0);
    CHECK(thief.tasks_stolen_in() > 0);
    CHECK(origin.tasks_stolen_out() > 0);
    CHECK(thief.tasks_stolen_in() == origin.tasks_stolen_out());

    std::printf("PASS test_cluster_remote_stealing: origin ran %llu, thief stole and ran %llu\n",
                (unsigned long long)origin_ran, (unsigned long long)thief_ran);

    origin.stop();
    thief.stop();
    coord.stop();
}

// With stealing disabled a node must keep all of its own work, which is the baseline
// the speedup claim is measured against.
static void test_no_stealing_keeps_work_local() {
    constexpr int NUM_TASKS = 100;
    constexpr uint32_t COST = 5000;

    g_executed.store(0);

    Coordinator coord(0);
    CHECK(coord.start());

    NodeConfig idle_cfg;
    idle_cfg.coordinator_port = coord.port();
    idle_cfg.num_workers      = 2;
    idle_cfg.enable_stealing  = false;

    NodeConfig origin_cfg;
    origin_cfg.coordinator_port = coord.port();
    origin_cfg.num_workers      = 2;
    origin_cfg.enable_stealing  = false;

    Node idle(idle_cfg);
    Node origin(origin_cfg);
    CHECK(idle.start());
    CHECK(origin.start());

    wait_until([&]() { return origin.peer_count() >= 1; }, 5000);

    for (int i = 0; i < NUM_TASKS; ++i) {
        origin.runtime().submit_portable(TaskType::SyntheticCompute, encode_cost(COST), COST);
    }
    origin.runtime().wait_all();

    CHECK(origin.runtime().local_executed() == (uint64_t)NUM_TASKS);
    CHECK(idle.runtime().local_executed() == 0);
    CHECK(origin.tasks_stolen_out() == 0);
    CHECK(g_executed.load() == NUM_TASKS);

    std::printf("PASS test_no_stealing_keeps_work_local: all %d tasks stayed on the origin\n",
                NUM_TASKS);

    origin.stop();
    idle.stop();
    coord.stop();
}

int main() {
    register_handlers();
    test_cluster_remote_stealing();
    test_no_stealing_keeps_work_local();
    std::printf("All tests passed.\n");
    return 0;
}
