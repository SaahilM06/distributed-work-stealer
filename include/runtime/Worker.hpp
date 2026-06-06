#pragma once

#include <atomic>
#include <functional>
#include <thread>

#include "runtime/WorkDeque.hpp"

class Worker {
public:
    Worker(int id,
           WorkDeque& queue,
           std::atomic<bool>& shutdown_flag,
           std::function<void()> on_complete);

    void start();
    void join();

private:
    void run();

    int                    id_;
    WorkDeque&             queue_;
    std::atomic<bool>&     shutdown_flag_;
    std::function<void()>  on_complete_;
    std::thread            thread_;
};
