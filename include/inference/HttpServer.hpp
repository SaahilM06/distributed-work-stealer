#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "net/Socket.hpp"

namespace inference {

struct HttpRequest {
    std::string method;
    std::string path;                          // path without the query string
    std::map<std::string, std::string> query;  // parsed ?a=1&b=2
    std::string body;
};

struct HttpResponse {
    int         status = 200;
    std::string content_type = "application/json";
    std::string body;
};

// A deliberately small HTTP/1.1 server: enough to accept inference jobs and report
// their status, with no external dependency. It reuses the same TcpSocket as the
// cluster layer, so there is one socket implementation in the project rather than two.
//
// Thread-per-connection, which is fine for a benchmark client but is not what you
// would ship in front of real traffic.
class HttpServer {
public:
    using Handler = std::function<HttpResponse(const HttpRequest&)>;

    HttpServer() = default;
    ~HttpServer();

    void route(const std::string& method, const std::string& path, Handler h);

    bool start(uint16_t port);
    void stop();

    uint16_t port() const { return listen_.local_port(); }

private:
    void accept_loop();
    void serve(TcpSocket conn);

    TcpSocket                listen_;
    std::atomic<bool>        running_{false};
    std::thread              accept_thread_;
    std::vector<std::thread> conn_threads_;
    std::mutex               conn_mutex_;

    std::map<std::string, Handler> routes_;   // key: "METHOD path"
};

} // namespace inference
