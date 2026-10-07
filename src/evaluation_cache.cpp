#include "evaluation_cache.hpp"

#include <algorithm>
#include <stdexcept>

EvaluationCache::EvaluationCache(std::size_t size, int history_depth)
    : m_history_depth(history_depth),
      m_entries(size),
      m_stripes(std::min<std::size_t>(size, 4096)) {
    if (size == 0) {
        throw std::invalid_argument("EvaluationCache : taille nulle");
    }
    if (history_depth < LEGACY_CACHE_HISTORY_DEPTH || history_depth > 7) {
        throw std::invalid_argument(
            "EvaluationCache : profondeur historique invalide");
    }
}

std::size_t EvaluationCache::index_for(
        const EvaluationCacheKey& key) const noexcept {
    return key.position_hash % m_entries.size();
}

std::size_t EvaluationCache::stripe_for(std::size_t index) const noexcept {
    return index % m_stripes.size();
}

TTProbe EvaluationCache::probe(const EvaluationCacheKey& key,
                               SearchTiming* timing) const {
    const std::size_t index = index_for(key);
    std::unique_lock<std::mutex> lock(
        m_stripes[stripe_for(index)], std::defer_lock);
    {
        PhaseTimer wait_timer(timing, SearchPhase::TTWait);
        lock.lock();
    }

    PhaseTimer access_timer(timing, SearchPhase::TTProbeStore);
    const TTEntry& entry = m_entries[index];
    TTProbe probe;
    if (entry.hash != key.position_hash || entry.policy_size == 0) {
        return probe;
    }
    if (m_history_depth != LEGACY_CACHE_HISTORY_DEPTH) {
        if (entry.half_move_clock != key.half_move_clock) {
            probe.status = TTProbeStatus::RULE50_REJECT;
            return probe;
        }
        if (entry.current_context_hash != key.current_context_hash) {
            probe.status = TTProbeStatus::CONTEXT_REJECT;
            return probe;
        }
        if (entry.history_hash != key.history_hash
            || entry.evaluation_hash != key.combined_hash) {
            probe.status = TTProbeStatus::HISTORY_REJECT;
            return probe;
        }
    }

    probe.status = TTProbeStatus::HIT;
    probe.value = entry.value;
    probe.policy_size = entry.policy_size;
    std::copy_n(entry.legal_policy.begin(), probe.policy_size,
                probe.legal_policy.begin());
    return probe;
}

void EvaluationCache::clear() {
    // Une bande a la fois, un seul verrou par bande : les index d'une meme
    // bande sont espaces de stripe_count, donc le parcours est direct.
    for (std::size_t stripe = 0; stripe < m_stripes.size(); ++stripe) {
        std::lock_guard<std::mutex> lock(m_stripes[stripe]);
        for (std::size_t index = stripe; index < m_entries.size();
             index += m_stripes.size()) {
            m_entries[index] = TTEntry{};
        }
    }
}

void EvaluationCache::store(const EvaluationCacheKey& key,
                            const std::vector<int>& legal_indices,
                            const float* policy, float value,
                            SearchTiming* timing) {
    // Tentative de mise en cache : une politique plus longue que la capacite
    // d'une entree est ignoree. L'arbre garde tous ses coups et la position
    // sera reevaluee ; une entree tronquee ne doit jamais etre servie comme un
    // hit complet.
    if (legal_indices.size() > static_cast<std::size_t>(TT_MAX_MOVES)) {
        return;
    }

    TTEntry replacement;
    {
        PhaseTimer prepare_timer(timing, SearchPhase::TTProbeStore);
        replacement.hash = key.position_hash;
        replacement.evaluation_hash = key.combined_hash;
        replacement.current_context_hash = key.current_context_hash;
        replacement.history_hash = key.history_hash;
        replacement.half_move_clock = key.half_move_clock;
        replacement.value = value;
        replacement.policy_size = static_cast<int>(legal_indices.size());
        for (int i = 0; i < replacement.policy_size; ++i) {
            const int move_index = legal_indices[i];
            replacement.legal_policy[i] = {move_index, policy[move_index]};
        }
    }

    const std::size_t index = index_for(key);
    std::unique_lock<std::mutex> lock(
        m_stripes[stripe_for(index)], std::defer_lock);
    {
        PhaseTimer wait_timer(timing, SearchPhase::TTWait);
        lock.lock();
    }
    {
        PhaseTimer publish_timer(timing, SearchPhase::TTProbeStore);
        m_entries[index] = replacement;
    }
}
