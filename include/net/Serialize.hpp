#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Explicit little-endian encoding rather than memcpy of structs: struct layout depends
// on the compiler's padding and the machine's byte order, so raw struct bytes are not
// a wire format. Every field is written byte by byte so both ends agree regardless.
class ByteWriter {
public:
    void u8(uint8_t v)  { buf_.push_back(v); }

    void u16(uint16_t v) {
        for (int i = 0; i < 2; ++i) buf_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    }

    void bytes(const void* p, std::size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        buf_.insert(buf_.end(), b, b + n);
    }

    // Length-prefixed so the reader knows where it ends.
    void str(const std::string& s) {
        u32(static_cast<uint32_t>(s.size()));
        bytes(s.data(), s.size());
    }
    void blob(const std::vector<uint8_t>& v) {
        u32(static_cast<uint32_t>(v.size()));
        bytes(v.data(), v.size());
    }

    const std::vector<uint8_t>& buf() const { return buf_; }
    std::vector<uint8_t>&       buf()       { return buf_; }

private:
    std::vector<uint8_t> buf_;
};

// Every read is bounds-checked and sets a sticky failure flag, so a truncated or
// malformed message can't run off the end of the buffer — this parses bytes that
// arrived over a socket, which must never be trusted to be well-formed.
class ByteReader {
public:
    ByteReader(const uint8_t* data, std::size_t size) : data_(data), size_(size) {}
    explicit ByteReader(const std::vector<uint8_t>& v) : data_(v.data()), size_(v.size()) {}

    uint8_t u8() {
        if (!need(1)) return 0;
        return data_[pos_++];
    }
    uint16_t u16() {
        if (!need(2)) return 0;
        uint16_t v = 0;
        for (int i = 0; i < 2; ++i) v |= static_cast<uint16_t>(data_[pos_ + i]) << (8 * i);
        pos_ += 2;
        return v;
    }
    uint32_t u32() {
        if (!need(4)) return 0;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(data_[pos_ + i]) << (8 * i);
        pos_ += 4;
        return v;
    }
    uint64_t u64() {
        if (!need(8)) return 0;
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(data_[pos_ + i]) << (8 * i);
        pos_ += 8;
        return v;
    }

    std::string str() {
        uint32_t n = u32();
        if (!need(n)) return {};
        std::string s(reinterpret_cast<const char*>(data_ + pos_), n);
        pos_ += n;
        return s;
    }
    std::vector<uint8_t> blob() {
        uint32_t n = u32();
        if (!need(n)) return {};
        std::vector<uint8_t> v(data_ + pos_, data_ + pos_ + n);
        pos_ += n;
        return v;
    }

    bool        ok() const        { return ok_; }
    std::size_t remaining() const { return ok_ ? size_ - pos_ : 0; }

private:
    bool need(std::size_t n) {
        if (!ok_ || pos_ + n > size_) { ok_ = false; return false; }
        return true;
    }

    const uint8_t* data_ = nullptr;
    std::size_t    size_ = 0;
    std::size_t    pos_  = 0;
    bool           ok_   = true;
};
