#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include "runtime/Task.hpp"

// Maps a TaskType to the code that runs it. Every node in the cluster registers the
// same table at startup, which is the trick that makes tasks shippable: the wire only
// has to carry a tag and a byte payload, not the code.
//
// Thread safety: register_handler() must be called before the Runtime starts its
// workers. After that the table is read-only and safe to read concurrently.
class TaskRegistry {
public:
    using Handler = std::function<void(const std::vector<uint8_t>& payload)>;

    static TaskRegistry& instance();

    void register_handler(TaskType type, Handler handler);
    bool has_handler(TaskType type) const;

    // Returns false if nothing is registered for `type` — which on a remote node means
    // the cluster's handler tables disagree, so the caller must not silently drop it.
    bool run(TaskType type, const std::vector<uint8_t>& payload) const;

    void clear();

private:
    TaskRegistry() = default;
    std::array<Handler, static_cast<std::size_t>(TaskType::Count)> handlers_{};
};

// The single place a task's work is invoked, whichever form it arrived in.
bool run_task(const Task& t);
