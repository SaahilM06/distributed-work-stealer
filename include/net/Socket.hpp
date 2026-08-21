#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "net/Protocol.hpp"

// Minimal blocking TCP socket. Move-only so a file descriptor has exactly one owner
// and is closed exactly once.
class TcpSocket {
public:
    TcpSocket() = default;
    explicit TcpSocket(int fd) : fd_(fd) {}
    ~TcpSocket();

    TcpSocket(TcpSocket&& other) noexcept;
    TcpSocket& operator=(TcpSocket&& other) noexcept;
    TcpSocket(const TcpSocket&)            = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    bool connect(const std::string& host, uint16_t port);
    bool listen(uint16_t port, int backlog = 64);
    TcpSocket accept();

    bool valid() const { return fd_.load(std::memory_order_acquire) >= 0; }
    int  fd() const    { return fd_.load(std::memory_order_acquire); }

    // Safe to call from another thread to unblock a thread parked in accept() or
    // recv() — that is how the accept loops here are shut down.
    void close();

    // Bounds the time a blocking recv can stall, so one unresponsive peer cannot pin a
    // thread forever. 0 disables the timeout.
    bool set_recv_timeout_ms(int ms);

    // send()/recv() may transfer fewer bytes than asked; these loop until done.
    bool send_all(const void* data, std::size_t n);
    bool recv_all(void* data, std::size_t n);

    // Single read of up to `n` bytes. Returns bytes read, 0 on clean close, -1 on
    // error. For protocols like HTTP where the length isn't known until the data has
    // been parsed, so recv_all() cannot be used.
    long recv_some(void* data, std::size_t n);

    bool send_msg(proto::MsgType type, const std::vector<uint8_t>& payload);
    bool recv_msg(proto::MsgType& out_type, std::vector<uint8_t>& out_payload);

    // Port actually bound, useful when listening on port 0 (kernel-assigned).
    uint16_t local_port() const;

private:
    // Atomic because one thread closes the socket to wake another that is blocked in
    // accept()/recv() on it. (A closed fd number can in principle be reused by another
    // socket before the blocked thread notices; the loops here treat any accept/recv
    // failure as "shutting down", which is sufficient for this usage.)
    std::atomic<int> fd_{-1};
};
