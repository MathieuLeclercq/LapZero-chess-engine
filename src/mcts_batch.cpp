// Noyau de recherche : boucle sequentielle historique et boucle batchee.
//
// Separe de mcts.cpp, qui depasse 520 lignes, sur le modele de
// mcts_observe.cpp. Voir docs/superpowers/specs/2026-09-11-mcts-batching-design.md

#include "mcts.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "search_terminal.hpp"

namespace {

void reserver_chemin(PathReservation& reservation, MCTSNode* node,
                     std::uint32_t units) {
    for (MCTSNode* current = node; current != nullptr; current = current->parent) {
        reservation.reserve(current, units);
    }
}

// Capture prise pendant que le plateau est sur la feuille. Le backup et
// l'expansion n'ont ensuite besoin que de cette capture et des pointeurs parent.
struct FeuilleCollectee {
    MCTSNode* node = nullptr;
    std::vector<int> legal_moves;
    EvaluationCacheKey key;
    PathReservation reservation;
};

}  // namespace

// Boucle sequentielle : une inference par simulation. Deplacee telle quelle
// depuis mcts_search, sans modification de l'ordre des operations, parce
// qu'elle sert de reference au test d'equivalence de la boucle batchee.

void MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations,
                      float c_puct, int batch_size,
                      SearchTiming* timing) {
    if (batch_size < 0) {
        throw std::invalid_argument("run_search : batch_size doit etre positif");
    }

    if (batch_size == 0) {
        for (int sim = 0; sim < simulations; sim++) {
            // Le garde restaure le plateau a la fin de la simulation, y compris
            // si l'evaluation de la feuille leve.
            BoardRollback rollback(board);
            auto [node, moves_played] = select_leaf(root, board, c_puct,
                                                    rollback, timing);

            const NodeState state =
                node->state.load(std::memory_order_acquire);
            if (state == NodeState::Terminal) {
                m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
                const float value = terminal_value_for(board);

                node->network_value = value;
                backup(node, value, timing);
                m_completed_simulations.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
            }

            if (state == NodeState::Unexpanded) {
                float value = expand_node_single(node, board, timing);
                backup(node, value, timing);
            }
            else if (state != NodeState::Expanded) {
                throw std::logic_error(
                    "run_search sequentiel : noeud Pending inattendu");
            }

            m_completed_simulations.fetch_add(
                1, std::memory_order_relaxed);
        }
        return;
    }

    std::vector<FeuilleCollectee> batch;
    std::vector<float> tensors;
    std::vector<float> policies;
    std::vector<float> values;
    std::vector<float> current_tensor;
    batch.reserve(static_cast<size_t>(batch_size));
    tensors.reserve(static_cast<size_t>(batch_size) * 119 * 64);
    current_tensor.reserve(119 * 64);

    int completed = 0;
    while (completed < simulations) {
        batch.clear();
        tensors.clear();

        while (static_cast<int>(batch.size()) < batch_size &&
               completed + static_cast<int>(batch.size()) < simulations) {
            // Une iteration de collecte : le garde restaure le plateau sur
            // toutes les sorties, y compris une erreur avant le retour.
            BoardRollback rollback(board);
            auto [node, moves_played] = select_leaf(root, board, c_puct,
                                                    rollback, timing);

            const NodeState state =
                node->state.load(std::memory_order_acquire);
            if (state == NodeState::Terminal) {
                m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
                const float value = terminal_value_for(board);
                node->network_value = value;
                backup(node, value, timing);
                completed++;
                m_completed_simulations.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
            }

            // select_leaf s'arrete ici lorsqu'une feuille deja collectee est
            // retrouvee. On evalue alors le lot partiel, comme LC0 apres
            // TryStartScoreUpdate, plutot que de forcer une nouvelle descente.
            if (state == NodeState::Pending) {
                break;
            }

            if (state != NodeState::Unexpanded) {
                throw std::logic_error(
                    "run_search batche : select_leaf a rendu un noeud Expanded");
            }

            EvaluationCacheKey key;
            {
                PhaseTimer timer(timing, SearchPhase::TensorKey);
                key = make_cache_key(board);
            }
            const TTProbe probe = probe_tt(key, timing);
            if (probe.status == TTProbeStatus::HIT) {
                throw std::logic_error(
                    "run_search : select_leaf a ignore un hit de TT");
            }
            record_tt_probe(probe.status);

            FeuilleCollectee leaf;
            leaf.node = node;
            if (!leaf.reservation.try_claim(node)) {
                break;
            }
            reserver_chemin(leaf.reservation, node,
                            static_cast<std::uint32_t>(m_tuning.virtual_loss));
            {
                PhaseTimer timer(timing, SearchPhase::TensorKey);
                leaf.legal_moves = board.getLegalMoveIndices();
            }
            leaf.key = key;

            if (leaf.legal_moves.empty()) {
                leaf.reservation.publish(NodeState::Terminal);
                const float terminal = board.isInCheck() ? -1.0f : 0.0f;
                node->network_value = terminal;
                backup(node, terminal, timing);
                leaf.reservation.release();
                completed++;
                m_completed_simulations.fetch_add(
                    1, std::memory_order_relaxed);
                continue;
            }

            {
                PhaseTimer timer(timing, SearchPhase::TensorKey);
                board.getAlphaZeroTensor(current_tensor);
            }
            {
                PhaseTimer timer(timing, SearchPhase::BatchAssembly);
                tensors.insert(tensors.end(), current_tensor.begin(),
                               current_tensor.end());
            }
            batch.push_back(std::move(leaf));
        }

        if (batch.empty()) continue;

        const int batch_count = static_cast<int>(batch.size());
        int eval_batch = batch_count;
        if (m_fixed_batch && batch_count < batch_size) {
            // Meme raison que dans les vagues : garder la forme du lot pour
            // eviter la re-planification d'ONNX Runtime.
            constexpr std::size_t TENSOR_SIZE = 119 * 64;
            eval_batch = batch_size;
            const std::size_t target =
                static_cast<std::size_t>(batch_size) * TENSOR_SIZE;
            tensors.reserve(target);
            const std::vector<float> dernier(
                tensors.end() - TENSOR_SIZE, tensors.end());
            while (tensors.size() < target) {
                tensors.insert(tensors.end(), dernier.begin(), dernier.end());
            }
        }

        m_nn_calls.fetch_add(batch_count, std::memory_order_relaxed);
        m_nn_batches.fetch_add(1, std::memory_order_relaxed);
        {
            PhaseTimer timer(timing, SearchPhase::Evaluator);
            m_evaluator->evaluate_batch(tensors, policies, values, eval_batch);
        }
        validate_network_output(policies, values, eval_batch);

        for (int i = 0; i < batch_count; ++i) {
            expand_and_backup_prepared(batch[i].node, batch[i].legal_moves,
                                       batch[i].key,
                                       policies.data() + static_cast<size_t>(i) * 4672,
                                       values[i], batch[i].reservation,
                                       timing);
            completed++;
            m_completed_simulations.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
}
