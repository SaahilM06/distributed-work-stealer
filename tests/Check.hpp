#pragma once

#include <cstdio>
#include <cstdlib>

// assert() compiles to nothing when NDEBUG is defined, which CMake does by default for
// both Release and RelWithDebInfo. That silently deletes the condition — and with it any
// side effect inside it, so `assert(sock.connect(...))` means the connect never happens
// and the test passes while testing nothing.
//
// CHECK always evaluates its argument and always aborts on failure, so a test binary is
// meaningful in every build type.
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n",               \
                         #cond, __FILE__, __LINE__);                             \
            std::fflush(stderr);                                                 \
            std::abort();                                                        \
        }                                                                        \
    } while (0)
