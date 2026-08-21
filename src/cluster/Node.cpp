#include "cluster/Node.hpp"
#include "runtime/TaskRegistry.hpp"

#include <algorithm>

const char* steal_policy_name(StealPolicy p) {
    switch (p) {
        case StealPolicy::None:      return "none";
        case StealPolicy::Random:    return "random";
        case StealPolicy::LoadAware: return "load-aware";
        case StealPolicy::Adaptive:  return "adaptive";
    }
    return "unknown";
}

bool parse_steal_policy(const std::string& s, StealPolicy& out) {
    if (s == "none")       { out = StealPolicy::None;      return true; }
    if (s == "random")     { out = StealPolicy::Random;    return true; }
    if (s == "load-aware" || s == "load") { out = StealPolicy::LoadAware; return true; }
    if (s == "adaptive")   { out = StealPolicy::Adaptive;  return true; }
    return false;
}

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
    reaper_thread_     = std::thread(&Node::reaper_loop, this);
    accept_thread_     = std::thread(&Node::accept_loop, this);
    coord_thread_      = std::thread(&Node::coordinator_loop, this);
    completion_thread_ = std::thread(&Node::completion_loop, this);
    if (cfg_.policy != StealPolicy::None) {
        steal_thread_ = std::thread(&Node::steal_loop, this);
    }

    std::printf("[node %u] listening on port %u  workers=%d  label=%s  policy=%s\n",
                node_id(), port(), cfg_.num_workers, cfg_.label.c_str(),
                steal_policy_name(cfg_.policy));
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
    if (reaper_thread_.joinable())     reaper_thread_.join();

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
    msg.host        = cfg_.advertise_host;
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
            if (p->info.node_id == info.node_id) {
                // Refresh the load figure; it is the input to victim selection and a
                // stale value would aim steals at the wrong peer.
                p->info.pending     = info.pending;
                p->info.label       = info.label;
                p->info.num_workers = info.num_workers;
                known = true;
                break;
            }
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

                TaskType preferred = static_cast<TaskType>(req.preferred_type);
                proto::StealResponseMsg resp;
                for (std::size_t i = 0; i < give; ++i) {
                    Task t;
                    if (!runtime_->take_portable(t, preferred)) break;
                    resp.tasks.push_back(std::move(t));
                }
                // Remember what we gave away and to whom. If the thief never reports
                // back, the reaper re-runs these locally.
                for (const Task& t : resp.tasks) {
                    track_outstanding(t, req.requester_node);
                }
                stolen_out_.fetch_add(resp.tasks.size(), std::memory_order_relaxed);
                // Tell the thief what is still queued here, so its next decision uses
                // a number from this instant rather than the last heartbeat.
                resp.victim_pending = runtime_->portable_available();
                conn.send_msg(proto::MsgType::StealResponse, proto::encode(resp));
                break;
            }
            case proto::MsgType::TaskResult: {
                proto::TaskResultMsg res;
                if (!proto::decode(payload, res)) break;
                // A task we handed out has finished elsewhere; close the books on it.
                // resolve_outstanding decides whether this is still owed: if the
                // reaper already gave up and re-ran the task locally, a late-arriving
                // result must be ignored or the task would be counted twice.
                resolve_outstanding(res.task_id);
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
            // Nothing available — back off rather than spin on the network. Under the
            // adaptive policy the wait grows as the success rate falls, so a cluster
            // with no spare work stops generating steal traffic instead of hammering
            // peers that have nothing to give.
            int backoff_ms = 1;
            if (cfg_.policy == StealPolicy::Adaptive) {
                // Exponential in *consecutive* failures, reset by any success, so a
                // quiet cluster stops generating traffic while a busy one is polled
                // aggressively.
                // Capped low on purpose: the cap is the worst-case delay before an
                // idle node notices that work has appeared somewhere. Backing off to
                // tens of milliseconds saves negligible CPU and makes a node useless
                // for any burst of work shorter than the backoff itself.
                uint32_t fails = consecutive_failures_.load(std::memory_order_relaxed);
                backoff_ms = 1;
                for (uint32_t i = 0; i < fails && backoff_ms < 8; ++i) backoff_ms *= 2;
            } else {
                backoff_ms = 2;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        }
    }
}

std::shared_ptr<Node::PeerLink> Node::select_victim() {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    if (peers_.empty()) return nullptr;

    switch (cfg_.policy) {
        case StealPolicy::None:
            return nullptr;

        case StealPolicy::Random: {
            // The classic work-stealing choice: no global knowledge needed, and it
            // spreads requests evenly so no single victim gets swamped.
            std::uniform_int_distribution<std::size_t> pick(0, peers_.size() - 1);
            return peers_[pick(rng_)];
        }

        case StealPolicy::LoadAware: {
            // Aim at whoever reported the most queued work on their last heartbeat.
            auto best = std::max_element(peers_.begin(), peers_.end(),
                [](const std::shared_ptr<PeerLink>& a, const std::shared_ptr<PeerLink>& b) {
                    return a->info.pending < b->info.pending;
                });
            if (best != peers_.end() && (*best)->info.pending > 0) return *best;
            // Nobody looks busy. Probe anyway: load figures only ever arrive as a
            // result of asking, so a policy that refuses to ask when it sees no load
            // can never discover that there is any.
            std::uniform_int_distribution<std::size_t> pick(0, peers_.size() - 1);
            return peers_[pick(rng_)];
        }

        case StealPolicy::Adaptive: {
            // Score = reported load, discounted by how often this peer has recently
            // come back empty. Heartbeat load is up to one interval stale, so a peer
            // that keeps saying "nothing here" is trusted less than its number claims.
            std::shared_ptr<PeerLink> best;
            double best_score = 0.0;
            for (const auto& p : peers_) {
                double load   = static_cast<double>(p->info.pending);
                double misses = static_cast<double>(p->misses.load(std::memory_order_relaxed));
                double score  = load / (1.0 + misses);
                if (score > best_score) {
                    best_score = score;
                    best       = p;
                }
            }
            if (best) return best;

            // Every peer looks idle. Probe one at random anyway — load figures lag, so
            // "everyone is idle" may simply be out of date.
            std::uniform_int_distribution<std::size_t> pick(0, peers_.size() - 1);
            return peers_[pick(rng_)];
        }
    }
    return nullptr;
}

bool Node::try_remote_steal() {
    std::shared_ptr<PeerLink> victim = select_victim();
    if (!victim) return false;

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
    req.preferred_type = static_cast<uint16_t>(preferred_type_);

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
    if (!proto::decode(payload, resp)) {
        victim->misses.fetch_add(1, std::memory_order_relaxed);
        steal_misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Record the victim's live depth whether or not it gave us anything — an empty
    // answer is still information, and it is fresher than any heartbeat.
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        victim->info.pending = resp.victim_pending;
    }

    if (resp.tasks.empty()) {
        victim->misses.fetch_add(1, std::memory_order_relaxed);
        steal_misses_.fetch_add(1, std::memory_order_relaxed);
        consecutive_failures_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    victim->misses.store(0, std::memory_order_relaxed);
    // Any success clears the backoff. Using a lifetime success rate instead makes the
    // backoff sticky: failures while the cluster is still starting up would suppress
    // stealing for the rest of the run, long after work became available.
    consecutive_failures_.store(0, std::memory_order_relaxed);
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

int64_t Node::now_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - origin_time_).count();
}

void Node::track_outstanding(const Task& t, uint32_t thief_node) {
    std::lock_guard<std::mutex> lock(outstanding_mutex_);
    OutstandingTask entry;
    entry.task       = t;
    entry.thief_node = thief_node;
    entry.sent_ms    = now_ms();
    outstanding_.emplace(t.task_id, std::move(entry));
}

void Node::resolve_outstanding(uint64_t task_id) {
    Task finished;
    {
        std::lock_guard<std::mutex> lock(outstanding_mutex_);
        auto it = outstanding_.find(task_id);
        if (it == outstanding_.end()) {
            // Already reaped and re-run locally — the local run has been (or will be)
            // counted, so completing it again here would overshoot `submitted`.
            return;
        }
        finished = std::move(it->second.task);
        outstanding_.erase(it);
    }
    runtime_->on_remote_task_complete(finished);
}

std::size_t Node::outstanding_count() const {
    std::lock_guard<std::mutex> lock(outstanding_mutex_);
    return outstanding_.size();
}

void Node::reap_outstanding() {
    std::vector<Task> lost;
    {
        std::lock_guard<std::mutex> lock(outstanding_mutex_);
        int64_t now = now_ms();
        for (auto it = outstanding_.begin(); it != outstanding_.end(); ) {
            if (now - it->second.sent_ms > cfg_.task_timeout_ms) {
                lost.push_back(std::move(it->second.task));
                it = outstanding_.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (Task& t : lost) {
        std::printf("[node %u] task %llu timed out on its thief — re-running locally\n",
                    node_id(), (unsigned long long)t.task_id);
        std::fflush(stdout);
        tasks_reassigned_.fetch_add(1, std::memory_order_relaxed);

        // Re-run here rather than putting it back in the portable pool: the peer that
        // just dropped it could otherwise steal it straight back and drop it again,
        // forever. Setting `fn` makes the task non-portable, while keeping `payload`
        // means a multi-stage job can still advance its chain when this run finishes —
        // dropping the payload would silently strand that request.
        Task rerun     = t;
        TaskType type  = t.type;
        std::vector<uint8_t> payload = t.payload;
        rerun.fn = [type, payload]() {
            TaskRegistry::instance().run(type, payload);
        };
        runtime_->submit_task(std::move(rerun));
        // The re-run is a *new* submission, so cancel the original's outstanding debt.
        // Passed with a cleared type so the observer does not advance the pipeline
        // twice: the re-run itself will advance it when it completes.
        Task settled;
        settled.task_id = t.task_id;
        runtime_->on_remote_task_complete(settled);
    }
}

void Node::reaper_loop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (!running_.load()) break;
        reap_outstanding();
    }
}

void Node::enqueue_completion(uint32_t origin_node, uint64_t task_id) {
    // Fault injection: behave like a node that took the work and then vanished.
    if (cfg_.drop_completions) return;
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
