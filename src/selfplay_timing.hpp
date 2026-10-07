#pragma once

#include <array>
#include <chrono>
#include <cstdint>

// Diagnostic de debit d'une generation self-play. Desactive par defaut :
// aucune horloge n'est lue et aucun compteur n'est incremente tant que
// l'instrumentation n'est pas activee. Toutes les durees sont en nanosecondes
// murales mesurees avec steady_clock.
//
// Deux niveaux de lecture :
//  - les phases de premier niveau sont disjointes : leur somme plus le residu
//    explicite reconstitue generation_wall_ns ;
//  - les sous-durees ONNX sont comptees separement et ne doivent pas etre
//    ajoutees aux phases qui les contiennent.
enum class SelfPlayPhase : std::size_t {
    InitialSlots,
    MoveManagement,
    Collection,
    AssemblyDispatch,
    BatchEvaluator,
    BatchValidation,
    BatchConsume,
    BatchFinalize,
    Count,
};

// Bornes d'histogramme : seau 0 pour 1 ligne, puis doublement jusqu'au seau 9
// qui regroupe 256 lignes et plus.
inline constexpr std::size_t SELFPLAY_BATCH_BUCKETS = 10;

inline std::size_t selfplay_batch_bucket(std::uint64_t rows) noexcept {
    std::size_t bucket = 0;
    std::uint64_t borne = 1;
    while (bucket + 1 < SELFPLAY_BATCH_BUCKETS && rows > borne) {
        ++bucket;
        borne *= 2;
    }
    return bucket;
}

struct SelfPlayTiming {
    static constexpr std::size_t PHASE_COUNT =
        static_cast<std::size_t>(SelfPlayPhase::Count);
    using Clock = std::chrono::steady_clock;

    bool enabled = false;
    int mode = 0;  // 0 desactive, 1 phases, 2 phases + detail par worker

    std::uint64_t generation_wall_ns = 0;
    std::array<std::uint64_t, PHASE_COUNT> phase_wall_ns{};
    // Residu signe par rapport au total de generation. Un residu negatif
    // signale un double comptage ou des frontieres incoherentes.
    std::int64_t generation_other_wall_ns = 0;

    // Sous-durees ONNX cumulees, appels de lot et appels unitaires des racines
    // confondus dans les deux premiers champs.
    std::uint64_t onnx_run_ns = 0;
    std::uint64_t onnx_softmax_ns = 0;
    std::uint64_t root_expansion_onnx_run_ns = 0;
    std::uint64_t root_expansion_onnx_softmax_ns = 0;

    // Compteurs de debit.
    std::uint64_t loop_turns = 0;
    std::uint64_t batch_calls = 0;        // appels physiques du gestionnaire
    std::uint64_t batch_rows = 0;         // lignes physiques de ces appels
    std::uint64_t unit_network_calls = 0; // racines ayant appele le reseau
    std::uint64_t unit_network_rows = 0;
    std::uint64_t max_batch_rows = 0;
    std::uint64_t batch_size_changes = 0;
    std::uint64_t deferred_turns = 0;     // tours avec un lot pret non envoye
    std::uint64_t deferred_wall_ns = 0;   // duree cumulee de ces tours
    std::uint64_t max_pending_age_ns = 0; // age maximal d'un lot en attente
    std::uint64_t root_expansions = 0;    // expansions de racine du gestionnaire
    std::uint64_t leaf_requests = 0;      // requetes logiques collectees
    // Descentes depuis les racines courantes, expansions unitaires exclues.
    std::uint64_t search_descents = 0;
    std::uint64_t search_depth_sum = 0;
    std::uint64_t search_depth_max = 0;
    std::uint64_t search_children_scored = 0; // un score UCB par enfant
    std::uint64_t completed_sims = 0;
    std::uint64_t no_network_sims = 0;    // terminees sans appel reseau
    std::uint64_t terminal_sims = 0;
    std::uint64_t tt_hits = 0;
    std::uint64_t tt_misses = 0;
    std::uint64_t slow_examples_saved = 0;
    std::array<std::uint64_t, SELFPLAY_BATCH_BUCKETS> batch_histogram{};

    // Vidange : du depart de la derniere partie du quota a la fin de la
    // generation. Plus aucune place ne se remplit, les lots retrecissent.
    std::uint64_t drain_wall_ns = 0;
    std::uint64_t drain_batch_calls = 0;
    std::uint64_t drain_batch_rows = 0;

    // Mode 2 : travail mural par iteration de collecte, somme et maximum sur
    // les workers. Ce n'est pas une duree murale de la phase.
    std::uint64_t worker_count = 0;
    std::uint64_t worker_busy_sum_ns = 0;
    std::uint64_t worker_busy_max_ns = 0;
};

// Chronometre de phase qui ne lit l'horloge que si l'instrumentation est
// activee. Un pointeur nul, cas du mode desactive, ne coute rien.
class SelfPlayPhaseTimer {
public:
    SelfPlayPhaseTimer(SelfPlayTiming* timing, SelfPlayPhase phase) noexcept
        : m_timing(timing), m_phase(phase) {
        if (m_timing != nullptr) {
            m_start = SelfPlayTiming::Clock::now();
        }
    }

    ~SelfPlayPhaseTimer() noexcept {
        if (m_timing == nullptr) return;
        const std::uint64_t duration_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                SelfPlayTiming::Clock::now() - m_start).count());
        m_timing->phase_wall_ns[static_cast<std::size_t>(m_phase)] +=
            duration_ns;
    }

    SelfPlayPhaseTimer(const SelfPlayPhaseTimer&) = delete;
    SelfPlayPhaseTimer& operator=(const SelfPlayPhaseTimer&) = delete;

private:
    SelfPlayTiming* m_timing;
    SelfPlayPhase m_phase;
    SelfPlayTiming::Clock::time_point m_start;
};
