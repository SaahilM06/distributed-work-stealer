// HydraRT inference server: a node in the cluster that also exposes an HTTP API.
//
//   POST /infer?model=text&seq=512      -> {"request_id":N}
//   GET  /status?id=N                   -> {"state":"done","latency_us":...}
//   GET  /stats                         -> cluster-visible counters
//
// Requests become four-stage task chains inside the runtime, and any stage can be
// stolen by another node in the cluster.

#include "cluster/Node.hpp"
#include "inference/HttpServer.hpp"
#include "inference/InferenceJob.hpp"
#include "inference/JobManager.hpp"
#include "metrics/Metrics.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>

using namespace inference;

static volatile std::sig_atomic_t g_stop = 0;
static void handle_signal(int) { g_stop = 1; }

// Requests whose cost varies over orders of magnitude. This is the workload the whole
// project exists to schedule: you cannot tell from the request count how much work a
// batch represents, so any up-front split is guaranteed to be wrong.
static Request random_request(std::mt19937& rng, uint64_t i) {
    std::uniform_int_distribution<int> pick(0, 99);
    Request req;

    if (pick(rng) < 50) {
        req.kind = ModelKind::ImageClassifier;
        // Mostly small images, occasionally a very large one.
        static const uint32_t sizes[] = {224 * 224, 224 * 224, 384 * 384, 512 * 512, 1024 * 1024};
        req.image_pixels = sizes[pick(rng) % 5];
    } else {
        req.kind = ModelKind::TextTransformer;
        // Sequence length drives a quadratic cost, so the tail here is severe.
        static const uint32_t lens[] = {64, 128, 128, 256, 512, 1024};
        req.sequence_length = lens[pick(rng) % 6];
    }
    req.batch_size = 1;
    (void)i;
    return req;
}

int main(int argc, char** argv) {
    NodeConfig cfg;
    uint16_t http_port    = 0;      // 0 = kernel picks
    int      bench_requests = 0;    // >0 runs a closed-loop benchmark instead of serving
    int      run_secs     = 30;

    for (int i = 1; i < argc; ++i) {
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (std::strcmp(argv[i], "--coordinator-port") == 0) {
            if (const char* v = next()) cfg.coordinator_port = static_cast<uint16_t>(std::atoi(v));
        } else if (std::strcmp(argv[i], "--coordinator-host") == 0) {
            if (const char* v = next()) cfg.coordinator_host = v;
        } else if (std::strcmp(argv[i], "--advertise-host") == 0) {
            if (const char* v = next()) cfg.advertise_host = v;
        } else if (std::strcmp(argv[i], "--port") == 0) {
            if (const char* v = next()) cfg.listen_port = static_cast<uint16_t>(std::atoi(v));
        } else if (std::strcmp(argv[i], "--http-port") == 0) {
            if (const char* v = next()) http_port = static_cast<uint16_t>(std::atoi(v));
        } else if (std::strcmp(argv[i], "--workers") == 0) {
            if (const char* v = next()) cfg.num_workers = std::atoi(v);
        } else if (std::strcmp(argv[i], "--label") == 0) {
            if (const char* v = next()) cfg.label = v;
        } else if (std::strcmp(argv[i], "--policy") == 0) {
            if (const char* v = next()) {
                if (!parse_steal_policy(v, cfg.policy)) {
                    std::fprintf(stderr, "unknown --policy '%s'\n", v);
                    return 2;
                }
            }
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            if (const char* v = next()) bench_requests = std::atoi(v);
        } else if (std::strcmp(argv[i], "--seconds") == 0) {
            if (const char* v = next()) run_secs = std::atoi(v);
        }
    }

    // Handlers must be registered before the runtime starts, and identically on every
    // node — that is what lets a stage stolen from another machine actually run here.
    JobManager::register_handlers(cfg.label);

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    Node node(cfg);
    if (cfg.label == "gpu") node.set_preferred_type(task_type_for(Stage::Infer));
    if (!node.start()) return 1;

    JobManager jobs(node.runtime(), cfg.label);

    // Let the coordinator hand out the peer list before any work is submitted.
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    // ── benchmark mode ──────────────────────────────────────────────────────
    if (bench_requests > 0) {
        std::mt19937 rng(12345);   // fixed seed: same workload across policies
        std::printf("[infer %u] label=%s policy=%s submitting %d requests\n",
                    node.node_id(), cfg.label.c_str(),
                    steal_policy_name(cfg.policy), bench_requests);
        std::fflush(stdout);

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_requests; ++i) {
            jobs.submit(random_request(rng, static_cast<uint64_t>(i)));
        }
        jobs.wait_all();
        double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        LatencyStats lat = compute_stats(jobs.request_latencies_ns());
        std::printf("[infer %u] BENCH policy=%s label=%s requests=%d "
                    "time=%.1f ms throughput=%.0f req/s "
                    "lat_ms p50=%.1f p95=%.1f p99=%.1f max=%.1f "
                    "executed=%llu stolen_in=%llu stolen_out=%llu reassigned=%llu\n",
                    node.node_id(), steal_policy_name(cfg.policy), cfg.label.c_str(),
                    bench_requests, elapsed_ms,
                    (bench_requests / elapsed_ms) * 1000.0,
                    lat.p50 / 1e6, lat.p95 / 1e6, lat.p99 / 1e6, lat.max / 1e6,
                    (unsigned long long)node.runtime().local_executed(),
                    (unsigned long long)node.tasks_stolen_in(),
                    (unsigned long long)node.tasks_stolen_out(),
                    (unsigned long long)node.tasks_reassigned());
        std::fflush(stdout);

        char path[256];
        std::snprintf(path, sizeof(path), "results/phase11_requests_%s.csv",
                      steal_policy_name(cfg.policy));
        jobs.dump_csv(path);

        node.stop();
        return 0;
    }

    // ── serving mode ────────────────────────────────────────────────────────
    HttpServer http;

    http.route("POST", "/infer", [&](const HttpRequest& req) {
        Request r;
        auto model = req.query.find("model");
        if (model != req.query.end() && model->second == "text") {
            r.kind = ModelKind::TextTransformer;
            auto seq = req.query.find("seq");
            if (seq != req.query.end()) r.sequence_length = std::atoi(seq->second.c_str());
        } else {
            r.kind = ModelKind::ImageClassifier;
            auto px = req.query.find("pixels");
            if (px != req.query.end()) r.image_pixels = std::atoi(px->second.c_str());
        }
        auto batch = req.query.find("batch");
        if (batch != req.query.end()) r.batch_size = std::atoi(batch->second.c_str());

        uint64_t id = jobs.submit(r);

        HttpResponse res;
        std::ostringstream body;
        body << "{\"request_id\":" << id
             << ",\"model\":\"" << model_kind_name(r.kind) << "\"}";
        res.body = body.str();
        return res;
    });

    http.route("GET", "/status", [&](const HttpRequest& req) {
        HttpResponse res;
        auto id_it = req.query.find("id");
        if (id_it == req.query.end()) {
            res.status = 400;
            res.body   = "{\"error\":\"missing id\"}";
            return res;
        }
        uint64_t  id = std::strtoull(id_it->second.c_str(), nullptr, 10);
        JobStatus st;
        if (!jobs.status(id, st)) {
            res.status = 404;
            res.body   = "{\"error\":\"unknown request\"}";
            return res;
        }
        const char* state = st.state == JobState::Done ? "done" : "running";
        std::ostringstream body;
        body << "{\"request_id\":" << st.request_id
             << ",\"state\":\"" << state << "\""
             << ",\"stage\":\"" << stage_name(st.stage) << "\""
             << ",\"latency_us\":" << (st.latency_ns() / 1000) << "}";
        res.body = body.str();
        return res;
    });

    http.route("GET", "/stats", [&](const HttpRequest&) {
        HttpResponse res;
        std::ostringstream body;
        body << "{\"node_id\":" << node.node_id()
             << ",\"label\":\"" << cfg.label << "\""
             << ",\"policy\":\"" << steal_policy_name(cfg.policy) << "\""
             << ",\"peers\":" << node.peer_count()
             << ",\"submitted\":" << jobs.submitted()
             << ",\"completed\":" << jobs.completed()
             << ",\"tasks_executed\":" << node.runtime().local_executed()
             << ",\"stolen_in\":" << node.tasks_stolen_in()
             << ",\"stolen_out\":" << node.tasks_stolen_out()
             << ",\"reassigned\":" << node.tasks_reassigned()
             << ",\"steal_success_rate\":" << node.steal_success_rate() << "}";
        res.body = body.str();
        return res;
    });

    if (!http.start(http_port)) {
        std::fprintf(stderr, "[infer] failed to bind HTTP port %u\n", http_port);
        node.stop();
        return 1;
    }

    std::printf("[infer %u] HTTP listening on http://127.0.0.1:%u  "
                "(POST /infer, GET /status?id=N, GET /stats)\n",
                node.node_id(), http.port());
    std::fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    while (!g_stop && std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start).count() < run_secs) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::printf("[infer %u] shutting down: %llu/%llu requests completed\n",
                node.node_id(),
                (unsigned long long)jobs.completed(),
                (unsigned long long)jobs.submitted());
    http.stop();
    node.stop();
    return 0;
}
