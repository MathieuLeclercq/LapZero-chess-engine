#pragma once

#include "chessboard.hpp"
#include "search_timing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

inline constexpr int TT_MAX_MOVES = 128;
inline constexpr int LEGACY_CACHE_HISTORY_DEPTH = -1;
inline constexpr int DEFAULT_CACHE_HISTORY_DEPTH = 0;

enum class TTProbeStatus {
    HIT,
    MISS,
    RULE50_REJECT,
    CONTEXT_REJECT,
    HISTORY_REJECT,
};

struct TTEntry {
    std::uint64_t hash = 0;
    std::uint64_t evaluation_hash = 0;
    std::uint64_t current_context_hash = 0;
    std::uint64_t history_hash = 0;
    std::uint16_t half_move_clock = 0;
    float value = 0.0f;
    int policy_size = 0;
    std::array<std::pair<int, float>, TT_MAX_MOVES> legal_policy{};
};

struct TTProbe {
    TTProbeStatus status = TTProbeStatus::MISS;
    float value = 0.0f;
    int policy_size = 0;
    std::array<std::pair<int, float>, TT_MAX_MOVES> legal_policy{};
};

class EvaluationCache {
public:
    EvaluationCache(std::size_t size, int history_depth);

    TTProbe probe(const EvaluationCacheKey& key,
                  SearchTiming* timing = nullptr) const;
    void store(const EvaluationCacheKey& key,
               const std::vector<int>& legal_indices,
               const float* policy, float value,
               SearchTiming* timing = nullptr);

    // Remise a froid complete, uniquement autorisee au repos (aucune lecture
    // ni ecriture concurrente). Le pool de workers n'est pas touche.
    void clear();

    std::size_t size() const noexcept { return m_entries.size(); }
    std::size_t stripe_count() const noexcept { return m_stripes.size(); }

private:
    std::size_t index_for(const EvaluationCacheKey& key) const noexcept;
    std::size_t stripe_for(std::size_t index) const noexcept;

    int m_history_depth;
    std::vector<TTEntry> m_entries;
    mutable std::vector<std::mutex> m_stripes;
};
