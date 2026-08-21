#include "cluster/Coordinator.hpp"

#include <cstdio>

Coordinator::Coordinator(uint16_t port)
    : port_(port)
    , origin_(std::chrono::steady_clock::now())
{}

Coordinator::~Coordinator() {
    stop();
}

int64_t Coordinator::now_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - origin_).count();
}

bool Coordinator::start() {
    if (!listen_.listen(port_)) return false;
    running_.store(true);
    accept_thread_ = std::thread(&Coordinator::accept_loop, this);
    return true;
}

void Coordinator::stop() {
    if (!running_.exchange(false)) return;

    // Closing the listening socket makes the blocking accept() in accept_loop return,
    // which is how that thread learns to exit.
    listen_.close();
    if (accept_thread_.joinable()) accept_thread_.join();

    std::lock_guard<std::mutex> lock(conn_mutex_);
    for (std::thread& t : conn_threads_) {
        if (t.joinable()) t.join();
    }
    conn_threads_.clear();
}

void Coordinator::accept_loop() {
    while (running_.load()) {
        TcpSocket conn = listen_.accept();
        if (!conn.valid()) break;   // listener closed, or accept failed

        std::lock_guard<std::mutex> lock(conn_mutex_);
        conn_threads_.emplace_back(&Coordinator::serve_connection, this, std::move(conn));
    }
}

void Coordinator::serve_connection(TcpSocket conn) {
    // A node that dies mid-connection would otherwise pin this thread in recv forever.
    conn.set_recv_timeout_ms(2000);

    while (running_.load()) {
        proto::MsgType       type;
        std::vector<uint8_t> payload;
        if (!conn.recv_msg(type, payload)) {
            // Timeout or disconnect. Keep looping while running so a quiet but healthy
            // node isn't dropped; the loop exits when the coordinator shuts down.
            if (!conn.valid()) break;
            continue;
        }

        switch (type) {
            case proto::MsgType::Register: {
                proto::RegisterMsg msg;
                if (!proto::decode(payload, msg)) break;
                uint32_t id = register_node(msg);
                std::printf("[coordinator] node %u registered: %s:%u workers=%u label=%s\n",
                            id, msg.host.c_str(), msg.port, msg.num_workers,
                            msg.label.c_str());
                std::fflush(stdout);
                conn.send_msg(proto::MsgType::RegisterAck,
                              proto::encode(proto::RegisterAckMsg{id}));
                break;
            }
            case proto::MsgType::Heartbeat: {
                proto::HeartbeatMsg msg;
                if (!proto::decode(payload, msg)) break;
                touch_node(msg.node_id, msg.pending, msg.completed);
                conn.send_msg(proto::MsgType::NodeList, proto::encode(current_nodes()));
                break;
            }
            default:
                break;
        }
    }
}

uint32_t Coordinator::register_node(const proto::RegisterMsg& msg) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);

    NodeRecord rec;
    rec.info.node_id     = next_node_id_++;
    rec.info.host        = msg.host;
    rec.info.port        = msg.port;
    rec.info.num_workers = msg.num_workers;
    rec.info.label       = msg.label;
    rec.last_seen_ms     = now_ms();

    nodes_.push_back(std::move(rec));
    return nodes_.back().info.node_id;
}

void Coordinator::touch_node(uint32_t node_id, uint64_t pending, uint64_t completed) {
    (void)pending;
    (void)completed;
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    for (NodeRecord& rec : nodes_) {
        if (rec.info.node_id == node_id) {
            rec.last_seen_ms = now_ms();
            return;
        }
    }
}

proto::NodeListMsg Coordinator::current_nodes() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    int64_t now = now_ms();

    proto::NodeListMsg out;
    for (const NodeRecord& rec : nodes_) {
        if (now - rec.last_seen_ms <= kNodeTimeoutMs) {
            out.nodes.push_back(rec.info);
        }
    }
    return out;
}

std::size_t Coordinator::live_node_count() const {
    return current_nodes().nodes.size();
}
