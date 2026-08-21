#include "cluster/Coordinator.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static volatile std::sig_atomic_t g_stop = 0;

static void handle_signal(int) {
    g_stop = 1;
}

int main(int argc, char** argv) {
    uint16_t port     = 9000;
    int      run_secs = 0;   // 0 = run until signalled

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = static_cast<uint16_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            run_secs = std::atoi(argv[++i]);
        }
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    Coordinator coord(port);
    if (!coord.start()) {
        std::fprintf(stderr, "[coordinator] failed to listen on port %u\n", port);
        return 1;
    }
    std::printf("[coordinator] listening on port %u\n", coord.port());
    std::fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (run_secs > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= run_secs) break;
        }
    }

    std::printf("[coordinator] shutting down (%zu live nodes)\n", coord.live_node_count());
    coord.stop();
    return 0;
}
