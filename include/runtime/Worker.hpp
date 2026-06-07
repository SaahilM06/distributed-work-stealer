#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "runtime/WorkDeque.hpp"

class Worker {
public:
    Worker(int id,
           WorkDeque& queue,
           std::vector<WorkDeque*> all_queues,
           std::atomic<bool>& shutdown_flag,
           std::function<void()> on_complete);

    void start();
    void join();

private:
    void run();

    int                    id_;
    WorkDeque&             queue_;
    std::vector<WorkDeque*> all_queues_;
    std::atomic<bool>&     shutdown_flag_;
    std::function<void()>  on_complete_;
    std::thread            thread_;
};
