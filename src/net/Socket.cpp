#include "net/Socket.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

TcpSocket::~TcpSocket() {
    close();
}

TcpSocket::TcpSocket(TcpSocket&& other) noexcept
    : fd_(other.fd_.exchange(-1, std::memory_order_acq_rel)) {}

TcpSocket& TcpSocket::operator=(TcpSocket&& other) noexcept {
    if (this != &other) {
        close();
        fd_.store(other.fd_.exchange(-1, std::memory_order_acq_rel),
                  std::memory_order_release);
    }
    return *this;
}

void TcpSocket::close() {
    int fd = fd_.exchange(-1, std::memory_order_acq_rel);
    if (fd >= 0) {
        ::close(fd);
    }
}

// Writing to a socket whose peer has gone away raises SIGPIPE, which by default kills
// the process. We want the error reported through send()'s return value instead.
static void suppress_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#else
    (void)fd;
#endif
}

bool TcpSocket::connect(const std::string& host, uint16_t port) {
    close();

    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* res = nullptr;
    std::string port_str = std::to_string(port);
    if (::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 || res == nullptr) {
        return false;
    }

    bool connected = false;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        int fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        suppress_sigpipe(fd);
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            int on = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
            fd_.store(fd, std::memory_order_release);
            connected = true;
            break;
        }
        ::close(fd);
    }
    ::freeaddrinfo(res);
    return connected;
}

bool TcpSocket::listen(uint16_t port, int backlog) {
    close();

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    suppress_sigpipe(fd);

    // Without SO_REUSEADDR a restarted node fails to bind while the old socket sits in
    // TIME_WAIT — painful when restarting a cluster repeatedly during development.
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(port);

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, backlog) != 0) {
        ::close(fd);
        return false;
    }
    fd_.store(fd, std::memory_order_release);
    return true;
}

TcpSocket TcpSocket::accept() {
    int fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return TcpSocket();
    int client = ::accept(fd, nullptr, nullptr);
    if (client < 0) return TcpSocket();
    suppress_sigpipe(client);
    int on = 1;
    ::setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
    return TcpSocket(client);
}

uint16_t TcpSocket::local_port() const {
    int fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return 0;
    sockaddr_in addr{};
    socklen_t   len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) return 0;
    return ntohs(addr.sin_port);
}

bool TcpSocket::set_recv_timeout_ms(int ms) {
    int fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return false;
    timeval tv{};
    tv.tv_sec  = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    return ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
}

bool TcpSocket::send_all(const void* data, std::size_t n) {
    int fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return false;
    const uint8_t* p    = static_cast<const uint8_t*>(data);
    std::size_t    sent = 0;
    while (sent < n) {
        ssize_t k = ::send(fd, p + sent, n - sent, 0);
        if (k > 0) {
            sent += static_cast<std::size_t>(k);
            continue;
        }
        if (k < 0 && errno == EINTR) continue;  // interrupted, not an error
        return false;
    }
    return true;
}

bool TcpSocket::recv_all(void* data, std::size_t n) {
    int fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return false;
    uint8_t*    p    = static_cast<uint8_t*>(data);
    std::size_t got  = 0;
    while (got < n) {
        ssize_t k = ::recv(fd, p + got, n - got, 0);
        if (k > 0) {
            got += static_cast<std::size_t>(k);
            continue;
        }
        if (k == 0) return false;               // peer closed cleanly
        if (errno == EINTR) continue;
        return false;
    }
    return true;
}

long TcpSocket::recv_some(void* data, std::size_t n) {
    int fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return -1;
    for (;;) {
        ssize_t k = ::recv(fd, data, n, 0);
        if (k < 0 && errno == EINTR) continue;
        return static_cast<long>(k);
    }
}

bool TcpSocket::send_msg(proto::MsgType type, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> framed = proto::frame(type, payload);
    return send_all(framed.data(), framed.size());
}

bool TcpSocket::recv_msg(proto::MsgType& out_type, std::vector<uint8_t>& out_payload) {
    uint8_t header[proto::kHeaderSize];
    if (!recv_all(header, sizeof(header))) return false;

    uint32_t len = 0;
    if (!proto::parse_header(header, sizeof(header), out_type, len)) return false;

    out_payload.resize(len);
    if (len > 0 && !recv_all(out_payload.data(), len)) return false;
    return true;
}
