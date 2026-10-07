#include "search_executor.hpp"

#include <stdexcept>

SearchExecutor::~SearchExecutor() {
    shutdown();
}

void SearchExecutor::run(
        std::size_t active_workers,
        const std::function<void(std::size_t)>& job) {
    std::unique_lock<std::mutex> run_lock(m_run_mutex);
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_shutdown) {
        throw std::logic_error("SearchExecutor est arrete");
    }
    if (!job) {
        throw std::invalid_argument("SearchExecutor : job vide");
    }
    if (active_workers == 0) {
        return;
    }

    // Le pool ne grandit qu'au repos, avant de publier le prochain job.
    try {
        for (std::size_t id = m_workers.size(); id < active_workers; ++id) {
            m_workers.emplace_back([this, id]() { worker_loop(id); });
        }
    }
    catch (...) {
        // Une creation partielle ne doit laisser aucun thread detache ni
        // endormi sur un objet dont la construction du pool a echoue.
        m_shutdown = true;
        lock.unlock();
        m_work_cv.notify_all();
        for (std::thread& worker : m_workers) {
            if (worker.joinable()) worker.join();
        }
        m_workers.clear();
        throw;
    }

    m_job = job;
    m_exception = nullptr;
    m_active_workers = active_workers;
    m_remaining = active_workers;
    ++m_generation;

    lock.unlock();
    m_work_cv.notify_all();
    lock.lock();
    m_done_cv.wait(lock, [this]() { return m_remaining == 0; });

    const std::exception_ptr exception = m_exception;
    m_job = {};
    m_active_workers = 0;
    lock.unlock();
    run_lock.unlock();

    if (exception) {
        std::rethrow_exception(exception);
    }
}

void SearchExecutor::shutdown() noexcept {
    std::unique_lock<std::mutex> run_lock(m_run_mutex);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_shutdown) {
            return;
        }
        m_shutdown = true;
    }
    m_work_cv.notify_all();

    for (std::thread& worker : m_workers) {
        if (worker.joinable()) worker.join();
    }
    m_workers.clear();
}

void SearchExecutor::worker_loop(std::size_t worker_id) noexcept {
    std::uint64_t observed_generation = 0;

    while (true) {
        const std::function<void(std::size_t)>* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_work_cv.wait(lock, [this, worker_id, observed_generation]() {
                return m_shutdown ||
                    (m_generation != observed_generation &&
                     worker_id < m_active_workers);
            });
            if (m_shutdown) {
                return;
            }

            observed_generation = m_generation;
            job = &m_job;
        }

        std::exception_ptr exception;
        try {
            (*job)(worker_id);
        }
        catch (...) {
            exception = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (exception && !m_exception) {
                m_exception = exception;
            }
            --m_remaining;
            if (m_remaining == 0) {
                m_done_cv.notify_one();
            }
        }
    }
}
