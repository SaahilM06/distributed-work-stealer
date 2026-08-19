#pragma once

#include <cstdint>
#include <functional>



enum class TaskType {
    Generic,          
    SyntheticCompute, 
    MandelbrotTile,   
    MergeSortRange,   
    MatrixBlock,      
};

struct Task {
    uint64_t task_id   = 0;
    uint64_t parent_id = 0;
    uint32_t depth     = 0;
    uint32_t cost_hint = 1;

    // Stamped by Runtime::submit() when tracing is on; the worker that runs this task
    // subtracts it from its start time to get queue wait. 0 when tracing is off.
    uint64_t submit_ns = 0;

    TaskType type = TaskType::Generic;

    std::function<void()> fn;
};
