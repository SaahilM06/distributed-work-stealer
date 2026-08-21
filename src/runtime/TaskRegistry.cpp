#include "runtime/TaskRegistry.hpp"

TaskRegistry& TaskRegistry::instance() {
    static TaskRegistry registry;
    return registry;
}

void TaskRegistry::register_handler(TaskType type, Handler handler) {
    if (type >= TaskType::Count) return;
    handlers_[static_cast<std::size_t>(type)] = std::move(handler);
}

bool TaskRegistry::has_handler(TaskType type) const {
    if (type >= TaskType::Count) return false;
    return static_cast<bool>(handlers_[static_cast<std::size_t>(type)]);
}

bool TaskRegistry::run(TaskType type, const std::vector<uint8_t>& payload) const {
    if (type >= TaskType::Count) return false;
    const Handler& h = handlers_[static_cast<std::size_t>(type)];
    if (!h) return false;
    h(payload);
    return true;
}

void TaskRegistry::clear() {
    for (Handler& h : handlers_) h = nullptr;
}

bool run_task(const Task& t) {
    if (t.fn) {
        t.fn();
        return true;
    }
    return TaskRegistry::instance().run(t.type, t.payload);
}
