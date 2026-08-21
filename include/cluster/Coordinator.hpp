#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "net/Protocol.hpp"
#include "net/Socket.hpp"

// Cluster membership service. Nodes register to get an id and learn about each other.
//
// Deliberately request/response only: a node sends a heartbeat and the coordinator
// replies with the current node list. Nothing is ever pushed to a node, which avoids
// having to synchronise writes to many sockets from one thread and means a node that
// misses an update simply picks it up on its next heartbeat.
class Coordinator {
public:
    // Milliseconds without a heartbeat after which a node is presumed dead and dropped
    // from the list handed to peers. Full failure handling is Phase 10.
    static constexpr int64_t kNodeTimeoutMs = 5000;

    explicit Coordinator(uint16_t port);
    ~Coordinator();

    bool     start();
    void     stop();
    uint16_t port() const { return listen_.local_port(); }

    std::size_t live_node_count() const;

private:
    struct NodeRecord {
        proto::NodeInfo info;
        int64_t         last_seen_ms = 0;
    };

    void accept_loop();
    void serve_connection(TcpSocket conn);

    uint32_t             register_node(const proto::RegisterMsg& msg);
    void                 touch_node(uint32_t node_id, uint64_t pending, uint64_t completed);
    proto::NodeListMsg   current_nodes() const;
    int64_t              now_ms() const;

    TcpSocket                 listen_;
    uint16_t                  port_;
    std::atomic<bool>         running_{false};
    std::thread               accept_thread_;
    std::vector<std::thread>  conn_threads_;
    std::mutex                conn_mutex_;

    mutable std::mutex       nodes_mutex_;
    std::vector<NodeRecord>  nodes_;
    uint32_t                 next_node_id_ = 1;

    std::chrono::steady_clock::time_point origin_;
};
