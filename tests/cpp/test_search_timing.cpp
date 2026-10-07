#include "search_timing.hpp"
#include "test_support.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void burn_cycles() {
    volatile std::uint64_t value = 0;
    for (std::uint64_t i = 0; i < 100000; ++i) {
        value += i;
    }
}

void test_disabled_timer_is_inert() {
    SearchTiming timing;
    {
        PhaseTimer timer(&timing, SearchPhase::Selection);
        burn_cycles();
    }

    require_test(timing.phase_ns(SearchPhase::Selection) == 0,
                 "disabled timer recorded time");
}

void test_enabled_timer_records_and_accumulates() {
    SearchTiming timing;
    timing.enabled = true;
    {
        PhaseTimer timer(&timing, SearchPhase::Selection);
        burn_cycles();
    }
    const auto first = timing.phase_ns(SearchPhase::Selection);
    require_test(first > 0, "enabled timer recorded no time");

    {
        PhaseTimer timer(&timing, SearchPhase::Selection);
        burn_cycles();
    }
    require_test(timing.phase_ns(SearchPhase::Selection) > first,
                 "phase durations were not accumulated");
}

void test_merge_combines_worker_timings() {
    SearchTiming left;
    left.enabled = true;
    left.wall_ns = 100;
    left.add_elapsed(SearchPhase::Selection, 20);

    SearchTiming right;
    right.enabled = true;
    right.wall_ns = 300;
    right.add_elapsed(SearchPhase::Selection, 7);
    right.add_elapsed(SearchPhase::Evaluator, 11);

    left.merge_worker(right);

    require_test(left.wall_ns == 100,
                 "worker merge changed coordinator wall time");
    require_test(left.phase_ns(SearchPhase::Selection) == 27,
                 "selection durations were not merged");
    require_test(left.phase_ns(SearchPhase::Evaluator) == 11,
                 "evaluator duration was not merged");
}

}  // namespace

int main() {
    try {
        test_disabled_timer_is_inert();
        test_enabled_timer_records_and_accumulates();
        test_merge_combines_worker_timings();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
