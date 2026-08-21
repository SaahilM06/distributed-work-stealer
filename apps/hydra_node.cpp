#include "cluster/Node.hpp"
#include "net/Serialize.hpp"
#include "runtime/TaskRegistry.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// Set by SIGTERM/SIGINT so a node shut down by the launcher still prints its RESULT
// line instead of dying silently.
static volatile std::sig_atomic_t g_stop = 0;

static void handle_signal(int) {
    g_stop = 1;
}

// The payload format for SyntheticCompute. Written and read with explicit field
// encoding rather than memcpy of a struct, so it stays valid across machines.
struct SyntheticPayload {
    uint32_t iterations = 0;
};

static std::vector<uint8_t> encode_synthetic(uint32_t iterations) {
    ByteWriter w;
    w.u32(iterations);
    return w.buf();
}

static void synthetic_work(uint32_t iterations) {
    volatile uint64_t x = 0;
    for (uint32_t i = 0; i < iterations; ++i) x += i;
}

// Registered identically on every node — this is what makes a task shippable.
static void register_handlers() {
    TaskRegistry::instance().register_handler(
        TaskType::SyntheticCompute,
        [](const std::vector<uint8_t>& payload) {
            ByteReader r(payload);
            uint32_t iterations = r.u32();
            if (!r.ok()) return;
            synthetic_work(iterations);
        });
}

int main(int argc, char** argv) {
    NodeConfig cfg;
    int      submit_tasks = 0;
    uint32_t task_cost    = 20000;
    int      run_secs     = 10;

    for (int i = 1; i < argc; ++i) {
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (std::strcmp(argv[i], "--coordinator-port") == 0) {
            if (const char* v = next()) cfg.coordinator_port = static_cast<uint16_t>(std::atoi(v));
        } else if (std::strcmp(argv[i], "--port") == 0) {
            if (const char* v = next()) cfg.listen_port = static_cast<uint16_t>(std::atoi(v));
        } else if (std::strcmp(argv[i], "--workers") == 0) {
            if (const char* v = next()) cfg.num_workers = std::atoi(v);
        } else if (std::strcmp(argv[i], "--label") == 0) {
            if (const char* v = next()) cfg.label = v;
        } else if (std::strcmp(argv[i], "--submit") == 0) {
            if (const char* v = next()) submit_tasks = std::atoi(v);
        } else if (std::strcmp(argv[i], "--task-cost") == 0) {
            if (const char* v = next()) task_cost = static_cast<uint32_t>(std::atoi(v));
        } else if (std::strcmp(argv[i], "--seconds") == 0) {
            if (const char* v = next()) run_secs = std::atoi(v);
        } else if (std::strcmp(argv[i], "--no-stealing") == 0) {
            cfg.enable_stealing = false;
        }
    }

    register_handlers();

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    Node node(cfg);
    if (!node.start()) return 1;

    // Give the coordinator a moment to hand out the peer list, so a node that submits
    // work isn't the only one that knows the cluster exists.
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    auto start = std::chrono::steady_clock::now();

    if (submit_tasks > 0) {
        std::printf("[node %u] submitting %d portable tasks (cost=%u)\n",
                    node.node_id(), submit_tasks, task_cost);
        std::fflush(stdout);
        for (int i = 0; i < submit_tasks; ++i) {
            node.runtime().submit_portable(TaskType::SyntheticCompute,
                                           encode_synthetic(task_cost), task_cost);
        }
        node.runtime().wait_all();

        double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("[node %u] all %d tasks complete in %.1f ms (%.0f tasks/sec)\n",
                    node.node_id(), submit_tasks, elapsed_ms,
                    (submit_tasks / elapsed_ms) * 1000.0);
    } else {
        // A pure worker node: stay up so others can steal from us, until the time
        // limit expires or the launcher signals us to stop.
        while (!g_stop && std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - start).count() < run_secs) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // Tasks this node's own workers ran. Deliberately not metrics().completed(), which
    // also counts tasks handed to a thief — summing that across nodes double counts.
    uint64_t executed = node.runtime().local_executed();

    std::printf("[node %u] RESULT executed=%llu"
                " stolen_in=%llu stolen_out=%llu steal_requests=%llu peers=%zu\n",
                node.node_id(), (unsigned long long)executed,
                (unsigned long long)node.tasks_stolen_in(),
                (unsigned long long)node.tasks_stolen_out(),
                (unsigned long long)node.steal_requests_sent(),
                node.peer_count());
    std::fflush(stdout);

    node.stop();
    return 0;
}
