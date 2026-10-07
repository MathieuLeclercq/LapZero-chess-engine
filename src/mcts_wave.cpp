#include "mcts.hpp"

#include "search_children.hpp"
#include "search_executor.hpp"
#include "search_terminal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

bool is_rule_terminal(const Chessboard& board) {
    return board.checkThreefoldRepetition()
        || board.getHalfMoveClock() >= 100
        || board.checkInsufficientMaterial();
}

float terminal_value(Chessboard& board) {
    if (is_rule_terminal(board)) return terminal_value_for(board);
    return board.isInCheck() ? -1.0f : 0.0f;
}

bool acquire_slot(std::atomic<std::size_t>& available) {
    std::size_t current = available.load(std::memory_order_relaxed);
    while (current > 0) {
        if (available.compare_exchange_weak(
                current, current - 1,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

// Les resultats d'un WorkerContext portent des reservations de chemin. A
// toute sortie de collect_wave, y compris une exception pendant la fusion,
// les contextes doivent etre vides : la racine locale de mcts_search est
// detruite pendant la propagation, et un resultat abandonne garderait des
// pointeurs et des unites en vol sur des noeuds liberes.
class WaveResultsGuard {
public:
    WaveResultsGuard(std::vector<WorkerContext>& contexts,
                     std::size_t worker_count)
        : m_contexts(contexts), m_worker_count(worker_count) {}

    WaveResultsGuard(const WaveResultsGuard&) = delete;
    WaveResultsGuard& operator=(const WaveResultsGuard&) = delete;

    ~WaveResultsGuard() {
        // Contrat de SearchExecutor::run : aucun worker ne touche plus aux
        // resultats quand run rend la main, exception comprise.
        for (std::size_t id = 0; id < m_worker_count; ++id) {
            m_contexts[id].results.clear();
        }
    }

private:
    std::vector<WorkerContext>& m_contexts;
    std::size_t m_worker_count;
};

}  // namespace

LeafWork MCTS::collect_wave_leaf(
        MCTSNode* root, WorkerContext& context, float c_puct,
        const std::atomic<bool>& cancelled,
        const WaveTestHooks* hooks) {
    if (root == nullptr) {
        throw std::invalid_argument("collect_wave_leaf : racine nulle");
    }

    LeafWork work;
    work.node = root;
    work.reservation.reserve(
        root, static_cast<std::uint32_t>(m_tuning.virtual_loss));
    BoardRollback rollback(context.board);
    MCTSNode* node = root;

    const auto collision = [&]() -> LeafWork {
        work.kind = LeafKind::Collision;
        work.reservation.release();
        m_leaf_collisions.fetch_add(1, std::memory_order_relaxed);
        return std::move(work);
    };

    while (true) {
        if (cancelled.load(std::memory_order_relaxed)) {
            return collision();
        }

        NodeState state = node->state.load(std::memory_order_acquire);

        // Les nulles de regle ne consultent jamais la table. Le noeud est
        // propre a ce chemin d'arbre, donc un etat Expanded serait incoherent.
        if (is_rule_terminal(context.board)) {
            if (state == NodeState::Pending) return collision();
            if (state == NodeState::Unexpanded) {
                if (hooks && hooks->before_claim) hooks->before_claim(node);
                if (!work.reservation.try_claim(node)) return collision();
                work.reservation.publish(NodeState::Terminal);
            }
            else if (state == NodeState::Terminal) {
                work.known_terminal = true;
            }
            else if (state != NodeState::Terminal) {
                throw std::logic_error(
                    "noeud de nulle deja publie Expanded");
            }
            work.kind = LeafKind::Terminal;
            work.node = node;
            work.terminal_value = terminal_value(context.board);
            return work;
        }

        if (state == NodeState::Terminal) {
            work.kind = LeafKind::Terminal;
            work.node = node;
            work.terminal_value = terminal_value(context.board);
            work.known_terminal = true;
            return work;
        }
        if (state == NodeState::Pending) {
            return collision();
        }

        if (state == NodeState::Unexpanded) {
            if (hooks && hooks->before_claim) hooks->before_claim(node);
            if (!work.reservation.try_claim(node)) {
                return collision();
            }

            std::vector<int> legal_moves;
            {
                PhaseTimer timer(&context.timing, SearchPhase::TensorKey);
                legal_moves = context.board.getLegalMoveIndices();
            }
            if (legal_moves.empty()) {
                work.reservation.publish(NodeState::Terminal);
                work.kind = LeafKind::Terminal;
                work.node = node;
                work.terminal_value = terminal_value(context.board);
                return work;
            }

            EvaluationCacheKey key;
            {
                PhaseTimer timer(&context.timing, SearchPhase::TensorKey);
                key = make_cache_key(context.board);
            }
            const TTProbe probe = probe_tt(key, &context.timing);
            record_tt_probe(probe.status);

            if (probe.status == TTProbeStatus::HIT) {
                node->network_value = probe.value;
                auto children = make_children_from_probe(node, probe);
                if (hooks && hooks->before_tt_publish) {
                    hooks->before_tt_publish(node);
                }
                {
                    PhaseTimer timer(
                        &context.timing, SearchPhase::Expansion);
                    node->children.swap(children);
                    work.reservation.publish(NodeState::Expanded);
                }
                state = NodeState::Expanded;
            }
            else {
                work.kind = LeafKind::Network;
                work.node = node;
                work.key = key;
                work.legal_moves = std::move(legal_moves);
                {
                    PhaseTimer timer(
                        &context.timing, SearchPhase::TensorKey);
                    context.board.getAlphaZeroTensor(work.tensor);
                }
                return work;
            }
        }

        if (state != NodeState::Expanded || node->children.empty()) {
            throw std::logic_error(
                "collect_wave_leaf : noeud publie sans enfants");
        }

        PhaseTimer selection_timer(
            &context.timing, SearchPhase::Selection);
        float visited_policy_sum = 0.0f;
        for (const auto& child : node->children) {
            if (child.second->visit_count > 0) {
                visited_policy_sum += child.second->prior;
            }
        }
        const float fpu_reduction =
            m_tuning.fpu_reduction * std::sqrt(visited_policy_sum);
        const float parent_q = node->q_value();
        const float exploration_factor =
            c_puct * std::sqrt(static_cast<float>(node->visit_count));

        float best_score = -std::numeric_limits<float>::infinity();
        int best_move = -1;
        MCTSNode* best_child = nullptr;
        for (const auto& child : node->children) {
            const float score = child.second->ucb_score(
                exploration_factor, parent_q, fpu_reduction);
            if (score > best_score) {
                best_score = score;
                best_move = child.first;
                best_child = child.second.get();
            }
        }
        if (best_child == nullptr) {
            throw std::logic_error(
                "collect_wave_leaf : aucun enfant selectionnable");
        }

        // Le virtual loss devient visible avant de jouer le coup. Deux
        // workers peuvent encore choisir simultanement le meme enfant, le CAS
        // Pending reste donc la garantie de correction.
        work.reservation.reserve(
            best_child, static_cast<std::uint32_t>(m_tuning.virtual_loss));
        if (!apply_move_by_index(context.board, best_move)) {
            throw std::runtime_error(
                "collect_wave_leaf : application du coup impossible");
        }
        rollback.move_played();
        node = best_child;
        work.node = node;
    }
}

std::vector<LeafWork> MCTS::collect_wave(
        MCTSNode* root, float c_puct, std::size_t slots,
        std::size_t worker_count, SearchExecutor& executor,
        std::vector<WorkerContext>& contexts,
        const std::atomic<bool>& cancelled,
        const WaveTestHooks* hooks) {
    if (root == nullptr) {
        throw std::invalid_argument("collect_wave : racine nulle");
    }
    if (slots == 0 || worker_count == 0) {
        throw std::invalid_argument(
            "collect_wave : slots et workers doivent etre positifs");
    }
    if (contexts.size() < worker_count) {
        throw std::invalid_argument(
            "collect_wave : contextes workers insuffisants");
    }
    if (slots > (std::numeric_limits<std::size_t>::max() - worker_count) / 4) {
        throw std::overflow_error("collect_wave : budget trop grand");
    }

    for (std::size_t id = 0; id < worker_count; ++id) {
        contexts[id].results.clear();
    }

    std::atomic<std::size_t> available{slots};
    std::atomic<std::size_t> attempts{0};
    std::atomic<bool> worker_failed{false};
    const std::size_t max_attempts =
        static_cast<std::size_t>(m_tuning.collision_attempt_factor) * slots
        + worker_count;

    // Capacite maximale reservee avant la collecte : le risque d'allocation se
    // place avant l'acquisition des reservations, et les transferts suivants
    // ne peuvent plus allouer.
    std::vector<LeafWork> merged;
    merged.reserve(slots);
    WaveResultsGuard guard(contexts, worker_count);

    executor.run(worker_count, [&](std::size_t worker_id) {
        try {
            WorkerContext& context = contexts[worker_id];
            while (!cancelled.load(std::memory_order_relaxed)
                   && !worker_failed.load(std::memory_order_relaxed)) {
                if (!acquire_slot(available)) return;

                const std::size_t attempt =
                    attempts.fetch_add(1, std::memory_order_relaxed);
                if (attempt >= max_attempts) {
                    available.fetch_add(1, std::memory_order_relaxed);
                    return;
                }

                LeafWork work = collect_wave_leaf(
                    root, context, c_puct, cancelled, hooks);
                if (work.kind == LeafKind::Collision) {
                    available.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                context.results.push_back(std::move(work));
            }
        }
        catch (...) {
            worker_failed.store(true, std::memory_order_relaxed);
            throw;
        }
    });

    if (hooks && hooks->before_merge) {
        hooks->before_merge();
    }

    for (std::size_t id = 0; id < worker_count; ++id) {
        for (LeafWork& work : contexts[id].results) {
            merged.push_back(std::move(work));
            if (hooks && hooks->after_transfer) {
                hooks->after_transfer(merged.size() - 1);
            }
        }
        contexts[id].results.clear();
    }

    if (merged.empty() && !cancelled.load(std::memory_order_relaxed)) {
        throw std::logic_error(
            "collect_wave : aucune progression apres les collisions");
    }
    return merged;
}

void MCTS::run_search_waves(
        MCTSNode* root, const Chessboard& board, int simulations,
        float c_puct, int batch_size, int worker_count,
        SearchTiming* timing) {
    if (simulations == 0) return;
    if (batch_size <= 0 || worker_count <= 1) {
        throw std::invalid_argument(
            "run_search_waves : configuration multicoeur invalide");
    }

    if (!m_search_executor) {
        m_search_executor = std::make_unique<SearchExecutor>();
    }
    if (m_worker_contexts.size()
            < static_cast<std::size_t>(worker_count)) {
        m_worker_contexts.resize(static_cast<std::size_t>(worker_count));
    }

    {
        PhaseTimer copy_timer(timing, SearchPhase::BoardCopy);
        for (int id = 0; id < worker_count; ++id) {
            m_worker_contexts[static_cast<std::size_t>(id)].board = board;
        }
    }

    std::atomic<bool> cancelled{false};
    int completed = 0;
    while (completed < simulations) {
        for (int id = 0; id < worker_count; ++id) {
            WorkerContext& context =
                m_worker_contexts[static_cast<std::size_t>(id)];
            context.timing = SearchTiming{};
            context.timing.enabled = timing != nullptr && timing->enabled;
        }

        const std::size_t slots = static_cast<std::size_t>(
            std::min(batch_size, simulations - completed));
        m_waves.fetch_add(1, std::memory_order_relaxed);

        std::vector<LeafWork> work;
        try {
            PhaseTimer wait_timer(timing, SearchPhase::WorkerWait);
            work = collect_wave(
                root, c_puct, slots,
                static_cast<std::size_t>(worker_count),
                *m_search_executor, m_worker_contexts, cancelled,
                m_wave_test_hooks);
        }
        catch (...) {
            cancelled.store(true, std::memory_order_relaxed);
            throw;
        }

        if (timing != nullptr && timing->enabled) {
            for (int id = 0; id < worker_count; ++id) {
                timing->merge_worker(
                    m_worker_contexts[static_cast<std::size_t>(id)].timing);
            }
        }

        std::vector<float> batch_input;
        std::vector<float> policies;
        std::vector<float> values;
        std::size_t network_count = 0;
        {
            PhaseTimer assembly_timer(timing, SearchPhase::BatchAssembly);
            for (const LeafWork& leaf : work) {
                if (leaf.kind != LeafKind::Network) continue;
                batch_input.insert(batch_input.end(),
                                   leaf.tensor.begin(), leaf.tensor.end());
                ++network_count;
            }
        }

        int eval_batch = static_cast<int>(network_count);
        if (m_fixed_batch && network_count > 0) {
            // ONNX Runtime re-planifie son graphe a chaque changement de forme
            // de lot, ce qui quadruple le cout de session->Run. On duplique le
            // dernier tenseur pour garder la forme batch_size ; seules les
            // network_count premieres sorties sont utilisees.
            constexpr std::size_t TENSOR_SIZE = 119 * 64;
            eval_batch = batch_size;
            const std::size_t target =
                static_cast<std::size_t>(batch_size) * TENSOR_SIZE;
            if (batch_input.size() < target) {
                batch_input.reserve(target);
                const std::vector<float> dernier(
                    batch_input.end() - TENSOR_SIZE, batch_input.end());
                while (batch_input.size() < target) {
                    batch_input.insert(batch_input.end(),
                                       dernier.begin(), dernier.end());
                }
            }
        }

        if (network_count > 0) {
            m_nn_calls.fetch_add(network_count, std::memory_order_relaxed);
            m_nn_batches.fetch_add(1, std::memory_order_relaxed);
            {
                PhaseTimer evaluator_timer(timing, SearchPhase::Evaluator);
                m_evaluator->evaluate_batch(
                    batch_input, policies, values, eval_batch);
            }

            validate_network_output(policies, values, eval_batch);
        }

        std::size_t network_index = 0;
        for (LeafWork& leaf : work) {
            if (leaf.kind == LeafKind::Network) {
                expand_and_backup_prepared(
                    leaf.node, leaf.legal_moves, leaf.key,
                    policies.data() + network_index * 4672,
                    values[network_index], leaf.reservation, timing);
                ++network_index;
            }
            else if (leaf.kind == LeafKind::Terminal) {
                if (leaf.known_terminal) {
                    m_terminal_hits.fetch_add(
                        1, std::memory_order_relaxed);
                }
                leaf.node->network_value = leaf.terminal_value;
                backup(leaf.node, leaf.terminal_value, timing);
                leaf.reservation.release();
            }
            else {
                throw std::logic_error(
                    "run_search_waves : collision fusionnee");
            }

            ++completed;
            m_completed_simulations.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
}
