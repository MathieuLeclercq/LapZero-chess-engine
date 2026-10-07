#include "search_timing.hpp"

#include <chrono>

namespace {

std::size_t phase_index(SearchPhase phase) noexcept {
    return static_cast<std::size_t>(phase);
}

}  // namespace

std::uint64_t SearchTiming::phase_ns(SearchPhase phase) const noexcept {
    return elapsed_ns[phase_index(phase)];
}

void SearchTiming::add_elapsed(SearchPhase phase,
                               std::uint64_t duration_ns) noexcept {
    elapsed_ns[phase_index(phase)] += duration_ns;
}

void SearchTiming::merge_worker(const SearchTiming& worker) noexcept {
    for (std::size_t i = 0; i < PHASE_COUNT; ++i) {
        elapsed_ns[i] += worker.elapsed_ns[i];
    }
}

PhaseTimer::PhaseTimer(SearchTiming* timing, SearchPhase phase) noexcept
    : m_timing(timing != nullptr && timing->enabled ? timing : nullptr),
      m_phase(phase) {
    if (m_timing != nullptr) {
        m_start = Clock::now();
    }
}

PhaseTimer::~PhaseTimer() noexcept {
    if (m_timing == nullptr) {
        return;
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - m_start);
    m_timing->add_elapsed(m_phase,
        static_cast<std::uint64_t>(elapsed.count()));
}
