#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

enum class SearchPhase : std::size_t {
    Selection,
    TensorKey,
    TTProbeStore,
    TTWait,
    BoardCopy,
    BatchAssembly,
    Evaluator,
    Expansion,
    Backup,
    WorkerWait,
    Count,
};

// Compteurs cumules par worker, sans horloge ni partage entre threads.
struct alignas(64) SearchWorkCounters {
    std::uint64_t descents = 0;
    std::uint64_t depth_sum = 0;
    std::uint64_t depth_max = 0;
    std::uint64_t children_scored = 0;
};

struct SearchTiming {
    static constexpr std::size_t PHASE_COUNT =
        static_cast<std::size_t>(SearchPhase::Count);

    bool enabled = false;
    std::uint64_t wall_ns = 0;
    std::array<std::uint64_t, PHASE_COUNT> elapsed_ns{};

    std::uint64_t phase_ns(SearchPhase phase) const noexcept;
    void add_elapsed(SearchPhase phase, std::uint64_t duration_ns) noexcept;
    void merge_worker(const SearchTiming& worker) noexcept;
};

class PhaseTimer {
public:
    PhaseTimer(SearchTiming* timing, SearchPhase phase) noexcept;
    ~PhaseTimer() noexcept;

    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;

private:
    using Clock = std::chrono::steady_clock;

    SearchTiming* m_timing;
    SearchPhase m_phase;
    Clock::time_point m_start;
};
