#include "cluster/Node.hpp"
#include "runtime/TaskRegistry.hpp"

#include <chrono>
#include <cstdio>

Node::Node(NodeConfig cfg) : cfg_(std::move(cfg)) {}

Node::~Node() {
    stop();
}

bool Node::start() {
    if (!listen_.listen(cfg_.listen_port)) {
        std::fprintf(stderr, "[node] failed to listen on port %u\n", cfg_.listen_port);
        return false;
    }

    runtime_ = std::make_unique<Runtime>(cfg_.num_workers);

    if (!register_with_coordinator()) {
        std::fprintf(stderr, "[node] failed to register with coordinator at %s:%u\n",
                     cfg_.coordinator_host.c_str(), cfg_.coordinator_port);
        return false;
    }
    runtime_->set_node_id(node_id_.load());

    running_.store(true);
    accept_thread_     = std::thread(&Node::accept_loop, this);
    coord_thread_      = std::thread(&Node::coordinator_loop, this);
    completion_thread_ = std::thread(&Node::completion_loop, this);
    if (cfg_.enable_stealing) {
        steal_thread_ = std::thread(&Node::steal_loop, this);
    }

    std::printf("[node %u] listening on port %u  workers=%d  label=%s  stealing=%s\n",
                node_id(), port(), cfg_.num_workers, cfg_.label.c_str(),
                cfg_.enable_stealing ? "on" : "off");
    std::fflush(stdout);
    return true;
}

void Node::stop() {
    if (!running_.exchange(false)) return;

    listen_.close();          // unblocks accept_loop
    completions_cv_.notify_all();   // unblocks completion_loop's timed wait
    if (accept_thread_.joinable())     accept_thread_.join();
    if (coord_thread_.joinable())      coord_thread_.join();
    if (steal_thread_.joinable())      steal_thread_.join();
    if (completion_thread_.joinable()) completion_thread_.join();

    {
        std::lock_guard<std::mutex> lock(peer_threads_mutex_);
        for (std::thread& t : peer_threads_) {
            if (t.joinable()) t.join();
        }
        peer_threads_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_.clear();
    }
    coord_sock_.close();
    runtime_.reset();
}

bool Node::register_with_coordinator() {
    if (!coord_sock_.connect(cfg_.coordinator_host, cfg_.coordinator_port)) return false;
    coord_sock_.set_recv_timeout_ms(3000);

    proto::RegisterMsg msg;
    msg.host        = "127.0.0.1";
    msg.port        = port();
    msg.num_workers = static_cast<uint32_t>(cfg_.num_workers);
    msg.label       = cfg_.label;

    if (!coord_sock_.send_msg(proto::MsgType::Register, proto::encode(msg))) return false;

    proto::MsgType       type;
    std::vector<uint8_t> payload;
    if (!coord_sock_.recv_msg(type, payload) || type != proto::MsgType::RegisterAck) return false;

    proto::RegisterAckMsg ack;
    if (!proto::decode(payload, ack)) return false;

    node_id_.store(ack.node_id, std::memory_order_relaxed);
    return true;
}

void Node::coordinator_loop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.heartbeat_ms));
        if (!running_.load()) break;

        proto::HeartbeatMsg hb;
        hb.node_id   = node_id();
        hb.pending   = runtime_->pending();
        hb.completed = runtime_->metrics().completed();

        if (!coord_sock_.send_msg(proto::MsgType::Heartbeat, proto::encode(hb))) continue;

        proto::MsgType       type;
        std::vector<uint8_t> payload;
        if (!coord_sock_.recv_msg(type, payload)) continue;
        if (type != proto::MsgType::NodeList) continue;

        proto::NodeListMsg list;
        if (proto::decode(payload, list)) update_peers(list);
    }
}

void Node::update_peers(const proto::NodeListMsg& msg) {
    std::lock_guard<std::mutex> lock(peers_mutex_);

    uint32_t self = node_id();
    for (const proto::NodeInfo& info : msg.nodes) {
        if (info.node_id == self) continue;

        bool known = false;
        for (const auto& p : peers_) {
            if (p->info.node_id == info.node_id) { known = true; break; }
        }
        if (known) continue;

        auto link  = std::make_shared<PeerLink>();
        link->info = info;
        peers_.push_back(std::move(link));
    }
}

std::size_t Node::peer_count() const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    return peers_.size();
}

std::shared_ptr<Node::PeerLink> Node::peer_for(uint32_t node_id) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (const auto& p : peers_) {
        if (p->info.node_id == node_id) return p;
    }
    return nullptr;
}

// ── inbound: serving other nodes' requests ──────────────────────────────────

void Node::accept_loop() {
    while (running_.load()) {
        TcpSocket conn = listen_.accept();
        if (!conn.valid()) break;

        std::lock_guard<std::mutex> lock(peer_threads_mutex_);
        peer_threads_.emplace_back(&Node::serve_peer, this, std::move(conn));
    }
}

void Node::serve_peer(TcpSocket conn) {
    conn.set_recv_timeout_ms(1000);

    while (running_.load()) {
        proto::MsgType       type;
        std::vector<uint8_t> payload;
        if (!conn.recv_msg(type, payload)) {
            if (!conn.valid()) break;
            continue;   // recv timeout; check running_ and wait again
        }

        switch (type) {
            case proto::MsgType::StealRequest: {
                proto::StealRequestMsg req;
                if (!proto::decode(payload, req)) break;

                // Steal-half: give away up to half of what's queued, capped by what the
                // thief asked for. Handing over everything would just move the pile and
                // leave this node idle; handing over one task per round trip would never
                // shift enough load to matter, since each trip costs a network RTT.
                std::size_t available = runtime_->portable_available();
                std::size_t give      = available / 2;
                if (give == 0 && available > 0) give = 1;
                if (give > req.max_tasks) give = req.max_tasks;

                proto::StealResponseMsg resp;
                for (std::size_t i = 0; i < give; ++i) {
                    Task t;
                    if (!runtime_->take_portable(t)) break;
                    resp.tasks.push_back(std::move(t));
                }
                stolen_out_.fetch_add(resp.tasks.size(), std::memory_order_relaxed);
                conn.send_msg(proto::MsgType::StealResponse, proto::encode(resp));
                break;
            }
            case proto::MsgType::TaskResult: {
                proto::TaskResultMsg res;
                if (!proto::decode(payload, res)) break;
                // A task we handed out has finished elsewhere; close the books on it.
                runtime_->on_remote_task_complete();
                break;
            }
            default:
                break;
        }
    }
}

// ── outbound: stealing from other nodes ─────────────────────────────────────

void Node::steal_loop() {
    // Steal while there isn't enough queued to keep every local worker busy. Waiting
    // for pending to hit exactly zero would mean a node fetches one task, works it to
    // completion, then makes another full network round trip — never building up
    // enough backlog to keep its workers fed.
    const uint64_t low_water = static_cast<uint64_t>(cfg_.num_workers);

    while (running_.load()) {
        if (runtime_->pending() >= low_water) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        if (!try_remote_steal()) {
            // Nothing available anywhere — back off rather than spin on the network.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}

bool Node::try_remote_steal() {
    std::shared_ptr<PeerLink> victim;
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        if (peers_.empty()) return false;
        // Random victim selection: the standard choice for work stealing, since it
        // spreads load without any node needing global knowledge.
        std::uniform_int_distribution<std::size_t> pick(0, peers_.size() - 1);
        victim = peers_[pick(rng_)];
    }

    std::lock_guard<std::mutex> lock(victim->mutex);

    if (!victim->connected) {
        if (!victim->sock.connect(victim->info.host, victim->info.port)) return false;
        victim->sock.set_recv_timeout_ms(2000);
        victim->connected = true;
    }

    proto::StealRequestMsg req;
    req.requester_node = node_id();
    // Ask for enough to keep this node's workers busy for a while, so the cost of the
    // round trip is amortised over many tasks rather than paid per task.
    req.max_tasks      = static_cast<uint32_t>(cfg_.num_workers) * 4;

    steal_requests_.fetch_add(1, std::memory_order_relaxed);
    if (!victim->sock.send_msg(proto::MsgType::StealRequest, proto::encode(req))) {
        victim->connected = false;
        victim->sock.close();
        return false;
    }

    proto::MsgType       type;
    std::vector<uint8_t> payload;
    if (!victim->sock.recv_msg(type, payload) || type != proto::MsgType::StealResponse) {
        victim->connected = false;
        victim->sock.close();
        return false;
    }

    proto::StealResponseMsg resp;
    if (!proto::decode(payload, resp) || resp.tasks.empty()) return false;

    for (const Task& t : resp.tasks) {
        accept_stolen_task(t);
    }
    stolen_in_.fetch_add(resp.tasks.size(), std::memory_order_relaxed);
    return true;
}

void Node::accept_stolen_task(const Task& remote) {
    TaskType             type    = remote.type;
    std::vector<uint8_t> payload = remote.payload;
    uint32_t             origin  = remote.origin_node;
    uint64_t             task_id = remote.task_id;

    // Submitted as a local closure task rather than back into the portable pool: this
    // node now owns running it, and wrapping it this way both keeps it from being
    // re-stolen onward and guarantees the origin is told when it finishes.
    runtime_->submit([this, type, payload, origin, task_id]() {
        TaskRegistry::instance().run(type, payload);
        enqueue_completion(origin, task_id);
    });
}

void Node::enqueue_completion(uint32_t origin_node, uint64_t task_id) {
    {
        std::lock_guard<std::mutex> lock(completions_mutex_);
        completions_.emplace_back(origin_node, task_id);
    }
    completions_cv_.notify_one();
}

void Node::completion_loop() {
    // Completion notices are sent from here, never from the worker that finished the
    // task: a worker must not block on a socket.
    while (running_.load()) {
        std::deque<std::pair<uint32_t, uint64_t>> batch;
        {
            std::unique_lock<std::mutex> lock(completions_mutex_);
            // Timed wait so shutdown is still noticed when no completions arrive.
            completions_cv_.wait_for(lock, std::chrono::milliseconds(50), [this]() {
                return !completions_.empty() || !running_.load();
            });
            batch.swap(completions_);
        }

        // Drain everything that accumulated in one pass, taking each peer's lock once
        // per batch rather than once per task.
        while (!batch.empty()) {
            std::pair<uint32_t, uint64_t> item = batch.front();
            batch.pop_front();

            std::shared_ptr<PeerLink> origin = peer_for(item.first);
            if (!origin) continue;

            std::lock_guard<std::mutex> lock(origin->mutex);
            if (!origin->connected) {
                if (!origin->sock.connect(origin->info.host, origin->info.port)) continue;
                origin->sock.set_recv_timeout_ms(2000);
                origin->connected = true;
            }

            proto::TaskResultMsg res;
            res.task_id = item.second;
            res.status  = 0;
            if (!origin->sock.send_msg(proto::MsgType::TaskResult, proto::encode(res))) {
                origin->connected = false;
                origin->sock.close();
            }
        }
    }
}
