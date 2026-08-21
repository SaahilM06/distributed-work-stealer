#pragma once

#include <cstdint>
#include <functional>
#include <vector>

// Tags a task's work so a remote node can find the right handler for it. Every node
// registers an identical handler table (see TaskRegistry), which is what lets a task
// cross a process boundary as a tag plus bytes.
enum class TaskType : uint16_t {
    Generic = 0,      // closure-only; never leaves this process
    SyntheticCompute,
    MandelbrotTile,
    MergeSortRange,
    MatrixBlock,
    Count             // sentinel — keep last
};

struct Task {
    uint64_t task_id   = 0;
    uint64_t parent_id = 0;
    uint32_t depth     = 0;
    uint32_t cost_hint = 1;

    // Stamped by Runtime::submit() when tracing is on; the worker that runs this task
    // subtracts it from its start time to get queue wait. 0 when tracing is off.
    uint64_t submit_ns = 0;

    // Which node originally submitted this task. A node that executes a stolen task
    // reports completion back here so the origin's accounting stays balanced.
    uint32_t origin_node = 0;

    TaskType type = TaskType::Generic;

    // Two ways to carry work, and exactly one is used per task:
    //   fn      — a closure over this process's memory. Fast, local-only: a lambda
    //             captures pointers that mean nothing on another machine, and
    //             std::function is type-erased machine code that cannot be serialized.
    //   payload — flat bytes interpreted by the handler registered for `type`. Slower
    //             to build, but this is the only form that can cross the network.
    std::function<void()> fn;
    std::vector<uint8_t>  payload;

    // Can this task be shipped to another node?
    bool portable() const { return !fn && type != TaskType::Generic; }
};
