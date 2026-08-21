#include "inference/HttpServer.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace inference {

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::route(const std::string& method, const std::string& path, Handler h) {
    routes_[method + " " + path] = std::move(h);
}

bool HttpServer::start(uint16_t port) {
    if (!listen_.listen(port)) return false;
    running_.store(true);
    accept_thread_ = std::thread(&HttpServer::accept_loop, this);
    return true;
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
    listen_.close();                       // unblocks accept()
    if (accept_thread_.joinable()) accept_thread_.join();

    std::lock_guard<std::mutex> lock(conn_mutex_);
    for (std::thread& t : conn_threads_) {
        if (t.joinable()) t.join();
    }
    conn_threads_.clear();
}

void HttpServer::accept_loop() {
    while (running_.load()) {
        TcpSocket conn = listen_.accept();
        if (!conn.valid()) break;

        std::lock_guard<std::mutex> lock(conn_mutex_);
        conn_threads_.emplace_back(&HttpServer::serve, this, std::move(conn));
    }
}

static std::string url_decode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') {
            out.push_back(' ');
        } else if (s[i] == '%' && i + 2 < s.size()) {
            out.push_back(static_cast<char>(std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16)));
            i += 2;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

static void parse_target(const std::string& target, HttpRequest& req) {
    std::size_t q = target.find('?');
    if (q == std::string::npos) {
        req.path = target;
        return;
    }
    req.path = target.substr(0, q);

    std::string qs = target.substr(q + 1);
    std::size_t pos = 0;
    while (pos < qs.size()) {
        std::size_t amp = qs.find('&', pos);
        if (amp == std::string::npos) amp = qs.size();
        std::string pair = qs.substr(pos, amp - pos);
        std::size_t eq   = pair.find('=');
        if (eq != std::string::npos) {
            req.query[url_decode(pair.substr(0, eq))] = url_decode(pair.substr(eq + 1));
        }
        pos = amp + 1;
    }
}

void HttpServer::serve(TcpSocket conn) {
    conn.set_recv_timeout_ms(5000);

    // Read until the end of headers. Small requests only — this is a benchmark
    // endpoint, not a general-purpose server.
    std::string buf;
    char        chunk[1024];
    while (buf.find("\r\n\r\n") == std::string::npos && buf.size() < 64 * 1024) {
        long n = conn.recv_some(chunk, sizeof(chunk));
        if (n <= 0) return;
        buf.append(chunk, static_cast<std::size_t>(n));
    }

    std::size_t header_end = buf.find("\r\n\r\n");
    if (header_end == std::string::npos) return;

    HttpRequest req;
    {
        std::istringstream head(buf.substr(0, header_end));
        std::string        line;
        if (!std::getline(head, line)) return;
        std::istringstream request_line(line);
        std::string        target;
        request_line >> req.method >> target;
        parse_target(target, req);

        // Body, if the client sent a Content-Length.
        std::size_t content_length = 0;
        while (std::getline(head, line)) {
            if (line.size() > 15 &&
                (line.compare(0, 15, "Content-Length:") == 0 ||
                 line.compare(0, 15, "content-length:") == 0)) {
                content_length = static_cast<std::size_t>(std::atoi(line.c_str() + 15));
            }
        }
        std::string body = buf.substr(header_end + 4);
        while (body.size() < content_length) {
            long n = conn.recv_some(chunk, sizeof(chunk));
            if (n <= 0) break;
            body.append(chunk, static_cast<std::size_t>(n));
        }
        req.body = body;
    }

    HttpResponse res;
    auto it = routes_.find(req.method + " " + req.path);
    if (it == routes_.end()) {
        res.status = 404;
        res.body   = "{\"error\":\"not found\"}";
    } else {
        res = it->second(req);
    }

    const char* reason = res.status == 200 ? "OK"
                       : res.status == 404 ? "Not Found"
                       : res.status == 400 ? "Bad Request" : "Error";

    std::ostringstream out;
    out << "HTTP/1.1 " << res.status << " " << reason << "\r\n"
        << "Content-Type: " << res.content_type << "\r\n"
        << "Content-Length: " << res.body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << res.body;

    std::string payload = out.str();
    conn.send_all(payload.data(), payload.size());
}

} // namespace inference
