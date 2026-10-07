#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class SearchExecutor {
public:
    SearchExecutor() = default;
    ~SearchExecutor();

    SearchExecutor(const SearchExecutor&) = delete;
    SearchExecutor& operator=(const SearchExecutor&) = delete;

    void run(std::size_t active_workers,
             const std::function<void(std::size_t)>& job);
    void shutdown() noexcept;

private:
    void worker_loop(std::size_t worker_id) noexcept;

    // Serialise les appels du coordinateur. Les callbacks workers ne prennent
    // jamais ce verrou.
    std::mutex m_run_mutex;
    std::mutex m_mutex;
    std::condition_variable m_work_cv;
    std::condition_variable m_done_cv;
    std::vector<std::thread> m_workers;
    std::function<void(std::size_t)> m_job;
    std::exception_ptr m_exception;
    std::uint64_t m_generation = 0;
    std::size_t m_active_workers = 0;
    std::size_t m_remaining = 0;
    bool m_shutdown = false;
};
