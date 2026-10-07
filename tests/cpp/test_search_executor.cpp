#include "search_executor.hpp"
#include "test_support.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace {

using namespace std::chrono_literals;

void test_workers_are_persistent_and_inactive_workers_sleep() {
    SearchExecutor executor;
    std::array<std::thread::id, 8> first_ids{};
    std::array<int, 8> calls{};

    executor.run(8, [&](std::size_t id) {
        first_ids[id] = std::this_thread::get_id();
        ++calls[id];
    });
    executor.run(2, [&](std::size_t id) {
        require_test(std::this_thread::get_id() == first_ids[id],
                     "worker thread changed between runs");
        ++calls[id];
    });

    require_test(calls[0] == 2 && calls[1] == 2,
                 "active workers did not run twice");
    require_test(calls[2] == 1 && calls[7] == 1,
                 "inactive worker executed a job");

    executor.run(8, [&](std::size_t id) {
        require_test(std::this_thread::get_id() == first_ids[id],
                     "grown active set recreated a worker");
        ++calls[id];
    });
    require_test(calls[0] == 3 && calls[1] == 3 && calls[7] == 2,
                 "reactivated workers ran the wrong number of jobs");
}

void test_destructor_does_not_execute_extra_work() {
    std::atomic<int> calls{0};
    {
        SearchExecutor executor;
        executor.run(4, [&](std::size_t) {
            calls.fetch_add(1, std::memory_order_relaxed);
        });
    }
    require_test(calls.load(std::memory_order_relaxed) == 4,
                 "destructor executed additional work");
}

void test_pool_can_grow_at_rest() {
    SearchExecutor executor;
    std::array<std::thread::id, 8> ids{};

    executor.run(2, [&](std::size_t id) {
        ids[id] = std::this_thread::get_id();
    });
    const auto first_id = ids[0];
    const auto second_id = ids[1];

    executor.run(8, [&](std::size_t id) {
        if (id == 0) {
            require_test(std::this_thread::get_id() == first_id,
                         "first worker was recreated during growth");
        }
        if (id == 1) {
            require_test(std::this_thread::get_id() == second_id,
                         "second worker was recreated during growth");
        }
        ids[id] = std::this_thread::get_id();
    });

    std::unordered_set<std::thread::id> unique_ids(ids.begin(), ids.end());
    require_test(unique_ids.size() == ids.size(),
                 "pool growth did not create eight distinct workers");
}

void test_callback_capture_is_released_before_return() {
    SearchExecutor executor;
    auto token = std::make_shared<int>(42);
    std::weak_ptr<int> weak = token;

    executor.run(2, [token](std::size_t) {});
    token.reset();

    require_test(weak.expired(),
                 "executor retained the callback after run returned");
}

void test_exception_waits_for_workers_and_next_run_recovers() {
    SearchExecutor executor;
    std::mutex mutex;
    std::condition_variable cv;
    bool second_worker_waiting = false;
    bool release_second_worker = false;

    auto run_result = std::async(std::launch::async, [&]() {
        try {
            executor.run(2, [&](std::size_t id) {
                if (id == 0) {
                    throw std::runtime_error("worker failure");
                }
                std::unique_lock<std::mutex> lock(mutex);
                second_worker_waiting = true;
                cv.notify_one();
                cv.wait(lock, [&]() { return release_second_worker; });
            });
        }
        catch (const std::runtime_error&) {
            return true;
        }
        return false;
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        require_test(cv.wait_for(lock, 2s, [&]() {
                         return second_worker_waiting;
                     }),
                     "second worker did not start");
    }
    require_test(run_result.wait_for(0ms) == std::future_status::timeout,
                 "run returned before all workers finished");

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_second_worker = true;
    }
    cv.notify_one();
    require_test(run_result.wait_for(2s) == std::future_status::ready,
                 "run did not finish after worker release");
    require_test(run_result.get(), "worker exception was not rethrown");

    std::array<int, 2> recovery_calls{};
    executor.run(2, [&](std::size_t id) { ++recovery_calls[id]; });
    require_test(recovery_calls[0] == 1 && recovery_calls[1] == 1,
                 "executor did not recover after an exception");
}

void test_shutdown_is_idempotent_and_final() {
    SearchExecutor executor;
    executor.run(2, [](std::size_t) {});
    executor.shutdown();
    executor.shutdown();

    bool rejected = false;
    try {
        executor.run(1, [](std::size_t) {});
    }
    catch (const std::logic_error&) {
        rejected = true;
    }
    require_test(rejected, "run accepted work after shutdown");
}

}  // namespace

int main() {
    try {
        test_workers_are_persistent_and_inactive_workers_sleep();
        test_destructor_does_not_execute_extra_work();
        test_pool_can_grow_at_rest();
        test_callback_capture_is_released_before_return();
        test_exception_waits_for_workers_and_next_run_recovers();
        test_shutdown_is_idempotent_and_final();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
