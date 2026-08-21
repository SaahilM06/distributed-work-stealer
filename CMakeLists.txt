cmake_minimum_required(VERSION 3.20)
project(HydraRT VERSION 0.1.0 LANGUAGES CXX)

# ── Language standard ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ── Compiler flags ───────────────────────────────────────────────────────────
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unused-parameter
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:RelWithDebInfo>:-O2>
  $<$<CONFIG:RelWithDebInfo>:-g>
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-fsanitize=thread>
)

add_link_options(
  $<$<CONFIG:Debug>:-fsanitize=thread>
)

# ── Threads ──────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Core library: hydra_runtime ──────────────────────────────────────────────
# All reusable runtime + metrics code lives here.
# Apps and tests link against this target.
add_library(hydra_runtime STATIC
  src/runtime/WorkDeque.cpp
  src/runtime/TaskRegistry.cpp
  src/runtime/InjectionQueue.cpp
  src/runtime/Worker.cpp
  src/runtime/Runtime.cpp
  src/metrics/Metrics.cpp
  src/net/Protocol.cpp
  src/net/Socket.cpp
  src/cluster/Coordinator.cpp
  src/cluster/Node.cpp
)

target_include_directories(hydra_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hydra_runtime PUBLIC
  Threads::Threads
)

# ── App: hydra_bench ─────────────────────────────────────────────────────────
add_executable(hydra_bench apps/hydra_bench.cpp)
target_link_libraries(hydra_bench PRIVATE hydra_runtime)

# ── Cluster binaries ─────────────────────────────────────────────────────────
add_executable(hydra_coordinator apps/hydra_coordinator.cpp)
target_link_libraries(hydra_coordinator PRIVATE hydra_runtime)

add_executable(hydra_node apps/hydra_node.cpp)
target_link_libraries(hydra_node PRIVATE hydra_runtime)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_deque    tests/test_deque.cpp)
add_executable(test_runtime  tests/test_runtime.cpp)
add_executable(test_protocol tests/test_protocol.cpp)
add_executable(test_cluster  tests/test_cluster.cpp)

target_link_libraries(test_deque    PRIVATE hydra_runtime)
target_link_libraries(test_runtime  PRIVATE hydra_runtime)
target_link_libraries(test_protocol PRIVATE hydra_runtime)
target_link_libraries(test_cluster  PRIVATE hydra_runtime)

add_test(NAME deque    COMMAND test_deque)
add_test(NAME runtime  COMMAND test_runtime)
add_test(NAME protocol COMMAND test_protocol)
add_test(NAME cluster  COMMAND test_cluster)

# ── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Targets     : hydra_runtime | hydra_bench | test_deque | test_runtime")
