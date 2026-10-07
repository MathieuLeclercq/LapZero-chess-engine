#include "mcts.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <cstdint>

#include "search_children.hpp"
#include "search_executor.hpp"
#include "search_terminal.hpp"

namespace {

using NodeChildren =
    std::vector<std::pair<int, std::unique_ptr<MCTSNode>>>;

NodeChildren make_children_from_policy(
        MCTSNode* parent, const std::vector<int>& legal_indices,
        const float* policy) {
    // Toute la liste legale construit des enfants, sans plafond lie au cache :
    // une position a plus de 128 coups legaux ne doit pas perdre les suivants.
    const int policy_size = static_cast<int>(legal_indices.size());
    float sum_legal = 0.0f;
    for (int i = 0; i < policy_size; ++i) {
        sum_legal += policy[legal_indices[i]];
    }

    NodeChildren children;
    children.reserve(policy_size);
    const float uniform = 1.0f / static_cast<float>(policy_size);
    for (int i = 0; i < policy_size; ++i) {
        const int move_index = legal_indices[i];
        const float prior = sum_legal > 0.0f
            ? policy[move_index] / sum_legal
            : uniform;
        children.emplace_back(
            move_index,
            std::make_unique<MCTSNode>(prior, move_index, parent));
    }
    return children;
}

void reserve_path(PathReservation& reservation, MCTSNode* leaf,
                  std::uint32_t units) {
    for (MCTSNode* node = leaf; node != nullptr; node = node->parent) {
        reservation.reserve(node, units);
    }
}

void validate_search_request(int num_simulations, int batch_size,
                             int worker_count) {
    if (num_simulations < 0) {
        throw std::invalid_argument(
            "num_simulations doit etre positif");
    }
    if (batch_size < 0) {
        throw std::invalid_argument("batch_size doit etre positif");
    }
    if (worker_count < 1) {
        throw std::invalid_argument("worker_count doit etre au moins 1");
    }
    if (worker_count > 1 && batch_size == 0) {
        throw std::invalid_argument(
            "la recherche multicoeur exige un batch_size positif");
    }
}

}  // namespace


// ============================================================
//                     MCTSNode
// ============================================================

MCTSNode::MCTSNode(float prior, int move_idx, MCTSNode* parent)
    : prior(prior), move_idx(move_idx), parent(parent),
    visit_count(0), total_value(0.0f) {
}

float MCTSNode::ucb_score(float exploration_factor, float parent_q, float fpu_reduction) const {
    // Implémentation du FPU de LeelaChess0
    // Si noeud pas visité, on ne met pas sa Q value à 0,
    // mais on utilise celle du parent.
    const std::uint32_t in_flight =
        n_in_flight.load(std::memory_order_relaxed);
    float u = exploration_factor * prior
        / (1.0f + visit_count + in_flight);
    float exploitation = (visit_count == 0) ? (parent_q - fpu_reduction) : -q_value();
    return exploitation + u;
}

float MCTSNode::q_value() const {
    if (visit_count == 0) return 0.0f;
    return total_value / visit_count;
}

MCTSNode* MCTSNode::find_child(int idx) const {
    for (const auto& pair : children) {
        if (pair.first == idx) return pair.second.get();
    }
    return nullptr;
}

bool MCTSNode::has_child(int idx) const {
    return find_child(idx) != nullptr;
}

std::unique_ptr<MCTSNode> MCTSNode::extract_child(int idx) {
    for (auto it = children.begin(); it != children.end(); ++it) {
        if (it->first == idx) {
            auto result = std::move(it->second);
            children.erase(it);
            return result;
        }
    }
    return nullptr;
}


// ============================================================
//                     MCTS
// ============================================================
MCTS::MCTS(Evaluator* evaluator, size_t tt_size,
           int cache_history_depth) :
        m_cache(tt_size, cache_history_depth),
        m_cache_history_depth(cache_history_depth),
        m_evaluator(evaluator),
        m_noise_rng(std::random_device{}()) {
    m_eval_tensor.reserve(119 * 64);
    m_eval_policy.reserve(4672);
}

MCTS::~MCTS() {
    if (m_search_executor) {
        m_search_executor->shutdown();
    }
}

EvaluationCacheKey MCTS::make_cache_key(const Chessboard& board) const {
    if (m_cache_history_depth != LEGACY_CACHE_HISTORY_DEPTH) {
        return board.getEvaluationCacheKey(m_cache_history_depth);
    }

    EvaluationCacheKey key;
    key.position_hash = board.getZobristHash();
    key.current_context_hash = key.position_hash;
    key.history_hash = key.position_hash;
    key.combined_hash = key.position_hash;
    return key;
}

TTProbe MCTS::probe_tt(const EvaluationCacheKey& key,
                       SearchTiming* timing) const {
    return m_cache.probe(key, timing);
}

void MCTS::record_tt_probe(TTProbeStatus status) {
    if (status == TTProbeStatus::MISS) {
        m_tt_misses.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    m_tt_position_matches.fetch_add(1, std::memory_order_relaxed);
    if (status == TTProbeStatus::HIT) {
        m_tt_hits.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    m_tt_misses.fetch_add(1, std::memory_order_relaxed);
    if (status == TTProbeStatus::RULE50_REJECT) {
        m_tt_rule50_rejects.fetch_add(1, std::memory_order_relaxed);
    }
    else if (status == TTProbeStatus::CONTEXT_REJECT) {
        m_tt_context_rejects.fetch_add(1, std::memory_order_relaxed);
    }
    else {
        m_tt_history_rejects.fetch_add(1, std::memory_order_relaxed);
    }
}

void MCTS::store_tt(const EvaluationCacheKey& key,
                    const std::vector<int>& legal_indices,
                    const float* policy, float value,
                    SearchTiming* timing) {
    m_cache.store(key, legal_indices, policy, value, timing);
}

void MCTS::backup(MCTSNode* node, float value, SearchTiming* timing) {
    PhaseTimer timer(timing, SearchPhase::Backup);
    while (node != nullptr) {
        node->visit_count += 1;
        node->total_value += value;
        value = -value;
        node = node->parent;
    }
}

std::pair<MCTSNode*, int> MCTS::select_leaf(MCTSNode* root, Chessboard& board,
                                           float c_puct,
                                           BoardRollback& rollback,
                                           SearchTiming* timing,
                                           SearchWorkCounters* work) {
    MCTSNode* node = root;
    int moves_played = 0;

    while (true) {
        const NodeState state = node->state.load(std::memory_order_acquire);
        if (state == NodeState::Terminal || state == NodeState::Pending) {
            break;
        }

        // --- 1. EXPANSION PARESSEUSE ---
        if (state == NodeState::Unexpanded) {
            EvaluationCacheKey key;
            {
                PhaseTimer timer(timing, SearchPhase::TensorKey);
                key = make_cache_key(board);
            }
            const TTProbe probe = probe_tt(key, timing);

            if (probe.status == TTProbeStatus::HIT) {
                PathReservation publication;
                if (!publication.try_claim(node)) {
                    continue;
                }
                record_tt_probe(probe.status);
                node->network_value = probe.value;
                PhaseTimer timer(timing, SearchPhase::Expansion);
                NodeChildren children = make_children_from_probe(node, probe);
                node->children.swap(children);
                publication.publish(NodeState::Expanded);
            }
            else {
                break; // Vraie feuille, besoin du GPU
            }
        }

        if (node->state.load(std::memory_order_acquire)
                != NodeState::Expanded) {
            continue;
        }
        if (node->children.empty()) {
            throw std::logic_error(
                "select_leaf : noeud Expanded sans enfant");
        }

        if (work != nullptr) {
            work->children_scored += node->children.size();
        }
        PhaseTimer selection_timer(timing, SearchPhase::Selection);

        // --- 2. FPU ---
        float visited_policy_sum = 0.0f;
        for (const auto& pair : node->children) {
            if (pair.second->visit_count > 0) {
                visited_policy_sum += pair.second->prior;
            }
        }
        float fpu_reduction = m_tuning.fpu_reduction
            * std::sqrt(visited_policy_sum);

        // --- 3. SÉLECTION UCB ---
        float max_ucb = -1e9f;
        int best_move_idx = -1;
        float parent_q = node->q_value();
        float exploration_factor = c_puct * std::sqrt(static_cast<float>(node->visit_count));

        MCTSNode* best_child = nullptr;
        for (const auto& pair : node->children) {
            float score = pair.second->ucb_score(exploration_factor, parent_q, fpu_reduction);
            if (score > max_ucb) {
                max_ucb = score;
                best_move_idx = pair.first;
                best_child = pair.second.get(); // Descente
            }
        }

        if (best_move_idx == -1) break;

        if (!apply_move_by_index(board, best_move_idx)) {
            throw std::runtime_error("Problème lors de l'application du coup dans select_leaf");
        }
        rollback.move_played();

        node = best_child;
        moves_played++;

        if (board.checkThreefoldRepetition() ||
            board.getHalfMoveClock() >= 100 ||
            board.checkInsufficientMaterial()) {
            // Valeur initialisee avant de publier l'etat : le mat prime sur la
            // nulle de regle quand les deux se recouvrent.
            node->network_value = terminal_value_for(board);
            node->state.store(NodeState::Terminal,
                              std::memory_order_release);
            break;
        }
    }

    return { node, moves_played };
}

float MCTS::expand_node_single(MCTSNode* node, Chessboard& board,
                               SearchTiming* timing) {
    PathReservation publication;
    if (!publication.try_claim(node)) {
        const NodeState state = node->state.load(std::memory_order_acquire);
        if (state == NodeState::Terminal) {
            return terminal_value_for(board);
        }
        throw std::logic_error(
            "expand_node_single : noeud deja publie ou reserve");
    }

    if (board.checkThreefoldRepetition() || board.getHalfMoveClock() >= 100 || board.checkInsufficientMaterial()) {
        const float terminal = terminal_value_for(board);
        node->network_value = terminal;
        publication.publish(NodeState::Terminal);
        return terminal;
    }

    EvaluationCacheKey key;
    {
        PhaseTimer timer(timing, SearchPhase::TensorKey);
        key = make_cache_key(board);
    }
    const TTProbe probe = probe_tt(key, timing);
    record_tt_probe(probe.status);

    if (probe.status == TTProbeStatus::HIT) {
        node->network_value = probe.value;
        // Materialiser les enfants tout de suite. Sans cela, le noeud reste
        // Unexpanded et un bruit de racine demande juste apres ne trouve aucune
        // liste d'enfants : le self-play perdait le bruit de Dirichlet sur les
        // racines servies par la table, notamment les positions de depart.
        auto children = make_children_from_probe(node, probe);
        node->children.swap(children);
        publication.publish(NodeState::Expanded);
        return probe.value;
    }

    std::vector<int> legal_indices;
    {
        PhaseTimer timer(timing, SearchPhase::TensorKey);
        legal_indices = board.getLegalMoveIndices();
    }
    if (legal_indices.empty()) {
        publication.publish(NodeState::Terminal);
        const float terminal = board.isInCheck() ? -1.0f : 0.0f;
        node->network_value = terminal;
        return terminal;
    }

    {
        PhaseTimer timer(timing, SearchPhase::TensorKey);
        board.getAlphaZeroTensor(m_eval_tensor);
    }
    float value;
    m_nn_calls.fetch_add(1, std::memory_order_relaxed);
    m_nn_batches.fetch_add(1, std::memory_order_relaxed);
    {
        PhaseTimer timer(timing, SearchPhase::Evaluator);
        m_evaluator->evaluate(m_eval_tensor, m_eval_policy, value);
    }

    store_tt(key, legal_indices, m_eval_policy.data(), value, timing);

    {
        PhaseTimer timer(timing, SearchPhase::Expansion);
        NodeChildren children = make_children_from_policy(
            node, legal_indices, m_eval_policy.data());
        node->children.swap(children);
        publication.publish(NodeState::Expanded);
    }

    node->network_value = value;
    return value;
}

void MCTS::add_dirichlet_noise(MCTSNode* root, float epsilon) {
    if (root->children.empty()) return;
    std::gamma_distribution<float> gamma(0.3f, 1.0f);

    float sum_noise = 0.0f;
    std::vector<float> noise(root->children.size());
    for (size_t i = 0; i < root->children.size(); i++) {
        noise[i] = gamma(m_noise_rng);
        sum_noise += noise[i];
    }

    int i = 0;
    for (auto& pair : root->children) {
        float dirichlet = noise[i++] / sum_noise;
        pair.second->prior = (1.0f - epsilon) * pair.second->prior + epsilon * dirichlet;
    }
}

std::vector<float> MCTS::mcts_search(
        Chessboard& board, int num_simulations, float c_puct,
        bool add_dirichlet, int batch_size, int worker_count) {

    validate_search_request(num_simulations, batch_size, worker_count);

    std::lock_guard<std::mutex> lock(m_mutex);
    SearchTiming timing;
    timing.enabled = m_timing_enabled;
    const auto wall_start = timing.enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto finish_timing = [&]() noexcept {
        if (timing.enabled) {
            timing.wall_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wall_start).count());
        }
        m_last_timing = timing;
    };

    //if (board.getZobristHash() != board.computeZobristFromScratch()) {
    //    throw std::runtime_error("Erreur fatale : Desynchronisation du Zobrist Hash detectee !");
    //}

    try {
        std::unique_ptr<MCTSNode> root = std::make_unique<MCTSNode>(0.0f);
        expand_node_single(root.get(), board, &timing);

        if (add_dirichlet) {
            add_dirichlet_noise(root.get(), 0.12f);
        }

        if (worker_count == 1) {
            run_search(root.get(), board, num_simulations, c_puct, batch_size,
                       &timing);
        }
        else {
            run_search_waves(root.get(), board, num_simulations, c_puct,
                             batch_size, worker_count, &timing);
        }

        std::vector<float> pi(4672, 0.0f);
        float sum_visits = 0.0f;
        for (const auto& pair : root->children) {
            pi[pair.first] = static_cast<float>(pair.second->visit_count);
            sum_visits += pi[pair.first];
        }

        if (sum_visits > 0.0f) {
            for (float& prob : pi) {
                prob /= sum_visits;
            }
        }

        finish_timing();
        return pi;
    }
    catch (...) {
        finish_timing();
        throw;
    }
}

bool MCTS::apply_move_by_index(Chessboard& board, int index) {
    // Le decodage vit desormais dans Chessboard, seul endroit qui connait
    // l'encodage des index et l'orientation du camp au trait.
    const Move move = board.decodeMoveIndex(index);
    return board.movePiece(
        move.getOrigSquare().getFile(), move.getOrigSquare().getRank(),
        move.getDestSquare().getFile(), move.getDestSquare().getRank(),
        move.getPromotion(), false);
}


// ============================================================
//                     ANALYSE CONTINUE
// ============================================================

void MCTS::reset_analysis() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_analysis_root.reset();
}

void MCTS::clear_evaluation_cache() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cache.clear();
}

void MCTS::set_fixed_batch(bool enabled) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_fixed_batch = enabled;
}

bool MCTS::fixed_batch() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_fixed_batch;
}

void MCTS::set_tuning(const SearchTuning& tuning) {
    if (tuning.virtual_loss < 1 || tuning.virtual_loss > 32) {
        throw std::invalid_argument(
            "virtual_loss doit etre compris entre 1 et 32");
    }
    if (tuning.fpu_reduction < 0.0f || tuning.fpu_reduction > 10.0f) {
        throw std::invalid_argument(
            "fpu_reduction doit etre compris entre 0 et 10");
    }
    if (tuning.collision_attempt_factor < 1
        || tuning.collision_attempt_factor > 64) {
        throw std::invalid_argument(
            "collision_attempt_factor doit etre compris entre 1 et 64");
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_tuning = tuning;
}

SearchTuning MCTS::get_tuning() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_tuning;
}

void MCTS::update_root(int move_idx) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_analysis_root && m_analysis_root->has_child(move_idx)) {
        m_analysis_root = m_analysis_root->extract_child(move_idx);
        m_analysis_root->parent = nullptr;
    }
    else {
        m_analysis_root.reset();
    }
}

float MCTS::get_root_q() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_analysis_root) return 0.0f;
    return m_analysis_root->q_value();
}

void MCTS::step_analysis(
        Chessboard& board, int num_simulations, float c_puct,
        int batch_size, int worker_count) {
    validate_search_request(num_simulations, batch_size, worker_count);
    // Le verrou couvre desormais toute la duree de l'appel, au lieu d'etre pris
    // et relache a chaque simulation. C'est necessaire pour la boucle batchee,
    // qui detiendra des MCTSNode* bruts pendant l'inference : relacher le verrou
    // en cours de lot laisserait update_root detruire l'arbre sous ces
    // pointeurs. Sans consequence pratique depuis que parse_position appelle
    // stop_search() avant toute modification de l'arbre (uci.py).
    std::lock_guard<std::mutex> lock(m_mutex);
    SearchTiming timing;
    timing.enabled = m_timing_enabled;
    const auto wall_start = timing.enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto finish_timing = [&]() noexcept {
        if (timing.enabled) {
            timing.wall_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wall_start).count());
        }
        m_last_timing = timing;
    };

    try {
        if (!m_analysis_root) {
            m_analysis_root = std::make_unique<MCTSNode>(0.0f);
            expand_node_single(m_analysis_root.get(), board, &timing);
        }

        if (worker_count == 1) {
            run_search(m_analysis_root.get(), board, num_simulations, c_puct,
                       batch_size, &timing);
        }
        else {
            run_search_waves(m_analysis_root.get(), board, num_simulations,
                             c_puct, batch_size, worker_count, &timing);
        }
        finish_timing();
    }
    catch (...) {
        finish_timing();
        throw;
    }
}

std::vector<MoveStats> MCTS::get_analysis_results() const {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<MoveStats> results;
    if (!m_analysis_root) return results;

    for (const auto& pair : m_analysis_root->children) {
        MCTSNode* child = pair.second.get();
        if (child->visit_count > 0) {
            results.push_back({
                pair.first,
                child->visit_count,
                -child->q_value(),
                child->prior,
                -child->network_value
                });
        }
    }

    std::sort(results.begin(), results.end(), [](const MoveStats& a, const MoveStats& b) {
        return a.visits > b.visits;
        });

    return results;
}

MCTSNode* MCTS::advance_to_leaf(MCTSNode* root, Chessboard& board,
                                float c_puct, int& moves_played,
                                PathReservation& reservation,
                                SearchWorkCounters* work) {
    // Sur le chemin nominal, l'appelant garde le plateau sur la feuille pour
    // lire le tenseur, puis le restaure avec moves_played : release() lui
    // transfere ce compte. Toute sortie anticipee est restauree par le garde.
    BoardRollback rollback(board);
    auto [node, moves] = select_leaf(
        root, board, c_puct, rollback, nullptr, work);
    if (work != nullptr) {
        ++work->descents;
        work->depth_sum += static_cast<std::uint64_t>(moves);
        work->depth_max = std::max(
            work->depth_max, static_cast<std::uint64_t>(moves));
    }
    moves_played = moves;

    const NodeState state = node->state.load(std::memory_order_acquire);
    if (state == NodeState::Terminal) {
        const float value = terminal_value_for(board);
        node->network_value = value;
        backup(node, value);
        return nullptr;
    }
    if (state == NodeState::Pending) {
        return nullptr;
    }
    if (state != NodeState::Unexpanded) {
        throw std::logic_error(
            "advance_to_leaf : select_leaf a rendu un noeud Expanded");
    }

    // Vérification TT avant GPU
    const EvaluationCacheKey key = make_cache_key(board);
    const TTProbe probe = probe_tt(key);
    record_tt_probe(probe.status);

    if (probe.status == TTProbeStatus::HIT) {
        node->network_value = probe.value;
        backup(node, probe.value);
        return nullptr;
    }

    if (!reservation.try_claim(node)) {
        return nullptr;
    }
    reserve_path(reservation, node,
                 static_cast<std::uint32_t>(m_tuning.virtual_loss));

    moves_played = rollback.release();
    return node;
}

void MCTS::expand_and_backup(MCTSNode* leaf_node, Chessboard& board,
                             const float* policy, float value,
                             PathReservation& reservation) {

    std::vector<int> legal_indices = board.getLegalMoveIndices();
    if (legal_indices.empty()) {
        reservation.publish(NodeState::Terminal);
        const float terminal = board.isInCheck() ? -1.0f : 0.0f;
        leaf_node->network_value = terminal;
        backup(leaf_node, terminal);
        reservation.release();
        return;
    }

    expand_and_backup_prepared(
        leaf_node, legal_indices, make_cache_key(board), policy, value,
        reservation);
}

void MCTS::expand_and_backup_prepared(MCTSNode* leaf_node,
                                      const std::vector<int>& legal_indices,
                                      const EvaluationCacheKey& key,
                                      const float* policy, float value,
                                      PathReservation& reservation,
                                      SearchTiming* timing) {
    // Les feuilles terminales sont traitees pendant la descente, jamais ici :
    // legal_indices est donc non vide et le plateau n'est pas necessaire.
    store_tt(key, legal_indices, policy, value, timing);

    leaf_node->network_value = value;

    {
        PhaseTimer timer(timing, SearchPhase::Expansion);
        NodeChildren children = make_children_from_policy(
            leaf_node, legal_indices, policy);
        leaf_node->children.swap(children);
        reservation.publish(NodeState::Expanded);
    }

    backup(leaf_node, value, timing);
    reservation.release();
}
