#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "selfplay_manager.hpp"
#include <piece.hpp>

namespace {

EvaluatorTotals delta_totals(const EvaluatorTotals& before,
                             const EvaluatorTotals& after) {
    EvaluatorTotals delta;
    delta.run_calls = after.run_calls - before.run_calls;
    delta.evaluated_rows = after.evaluated_rows - before.evaluated_rows;
    delta.timing.run_ns = after.timing.run_ns - before.timing.run_ns;
    delta.timing.softmax_ns =
        after.timing.softmax_ns - before.timing.softmax_ns;
    return delta;
}

std::uint64_t elapsed_ns_since(SelfPlayTiming::Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            SelfPlayTiming::Clock::now() - start).count());
}

}  // namespace

SelfPlayManager::SelfPlayManager(
    Evaluator* evaluator,
    int num_concurrent_games,
    int slow_sims, int fast_sims, float slow_ratio,
    size_t tt_size,
    const std::string& puzzles_path,
    const std::string& start_positions_path)
    : m_evaluator(evaluator),
    m_num_concurrent_games(num_concurrent_games),
    m_slow_sims(slow_sims),
    m_fast_sims(fast_sims),
    m_slow_ratio(slow_ratio)
{
    if (num_concurrent_games <= 0) {
        throw std::invalid_argument(
            "SelfPlayManager : le nombre de places doit etre positif");
    }

    m_boards.resize(num_concurrent_games);
    m_roots.resize(num_concurrent_games);
    m_sims_completed.resize(num_concurrent_games, 0);
    m_is_waiting.resize(num_concurrent_games, false);
    m_slot_active.resize(num_concurrent_games, false);

    m_sims_target.resize(num_concurrent_games, 0);
    m_is_slow_move.resize(num_concurrent_games, false);
    m_pending_epsilon.resize(num_concurrent_games, 0.0f);
    m_forced_end_plies.resize(num_concurrent_games, 0);

    m_game_states.resize(num_concurrent_games);
    m_game_policies.resize(num_concurrent_games);

    m_batch_input.resize(num_concurrent_games * 119 * 64);

    // Le virtual loss n'est pas regle ici : en self-play, chaque arbre n'est
    // descendu qu'une fois par vague, donc n_in_flight n'a pas de lecteur entre
    // la reservation et sa liberation, et l'amplitude est inerte. Le bot et les
    // bancs, qui collectent plusieurs feuilles du meme arbre, la reglent a 2.
    m_shared_mcts = std::make_unique<MCTS>(m_evaluator, tt_size);

    m_tactical_boost.resize(num_concurrent_games, false);
    m_replayed_plies.resize(num_concurrent_games, 0);
    load_tactical_puzzles(puzzles_path);
    if (!start_positions_path.empty()) {
        load_start_positions(start_positions_path);
    }
}

void SelfPlayManager::demarrer_slot(int game_idx) {
    // Le quota est connu ici : aucune partie ne demarre avant generate_games.
    reset_game(game_idx);
    m_slot_active[game_idx] = true;
    m_stats.games_started++;
}

SelfPlayStats SelfPlayManager::get_stats() const {
    SelfPlayStats stats = m_stats;
    stats.active_slots = 0;
    for (char actif : m_slot_active) {
        if (actif) ++stats.active_slots;
    }
    return stats;
}

void SelfPlayManager::set_diagnostics_mode(int mode) {
    if (mode < 0 || mode > 2) {
        throw std::invalid_argument(
            "set_diagnostics_mode : mode attendu entre 0 et 2");
    }
    m_diagnostics_mode = mode;
}

int SelfPlayManager::get_diagnostics_mode() const {
    return m_diagnostics_mode;
}

SelfPlayTiming SelfPlayManager::get_timing() const {
    return m_timing;
}

void SelfPlayManager::set_fast_dirichlet_noise(bool enabled) {
    m_fast_dirichlet_noise = enabled;
}

float SelfPlayManager::epsilon_for_move(int game_idx) const {
    const std::size_t index = static_cast<std::size_t>(game_idx);
    if (m_tactical_boost[index]) {
        return TACTICAL_EPSILON;
    }
    if (!m_fast_dirichlet_noise && !m_is_slow_move[index]) {
        return 0.0f;
    }
    return NORMAL_EPSILON;
}

void SelfPlayManager::expand_root(int game_idx) {
    if (!m_timing.enabled) {
        m_shared_mcts->expand_node_single(
            m_roots[game_idx].get(), m_boards[game_idx]);
        return;
    }

    m_timing.root_expansions++;
    const EvaluatorTotals before = m_evaluator->diagnostic_totals();
    m_shared_mcts->expand_node_single(
        m_roots[game_idx].get(), m_boards[game_idx]);
    const EvaluatorTotals delta =
        delta_totals(before, m_evaluator->diagnostic_totals());
    if (delta.run_calls > 0) {
        m_timing.unit_network_calls += delta.run_calls;
        m_timing.unit_network_rows += delta.evaluated_rows;
        m_timing.root_expansion_onnx_run_ns += delta.timing.run_ns;
        m_timing.root_expansion_onnx_softmax_ns += delta.timing.softmax_ns;
    }
}

int SelfPlayManager::selfplay_plies(int game_idx) const {
    return static_cast<int>(m_boards[game_idx].getMoveHistory().size())
        - m_replayed_plies[game_idx];
}

bool SelfPlayManager::longueur_max_atteinte(int game_idx) const {
    return selfplay_plies(game_idx) >= MAX_SELFPLAY_PLIES;
}

void SelfPlayManager::reset_game(int game_idx) {
    m_boards[game_idx].clear();
    m_boards[game_idx].setAmnesiaMode(false);
    m_replayed_plies[game_idx] = 0;

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // --- POSITIONS DE DEPART IMPOSEES (bancs de diagnostic) ---
    if (!m_start_fens.empty()) {
        std::uniform_int_distribution<size_t> idx_dis(0, m_start_fens.size() - 1);
        m_boards[game_idx].loadFEN(m_start_fens[idx_dis(m_rng)]);
        m_tactical_boost[game_idx] = false;
    }
    // --- INJECTION DE PUZZLE (20% du temps) ---
    else if (!m_tactical_puzzles.empty() && dis(m_rng) < 0.2f) {
        std::uniform_int_distribution<size_t> idx_dis(0, m_tactical_puzzles.size() - 1);
        const TacticalPuzzle& puzzle = m_tactical_puzzles[idx_dis(m_rng)];

        m_boards[game_idx].loadFEN(puzzle.start_fen);

        // On rejoue les coups réels de la partie pour que m_boardHistory
        // contienne un historique authentique. Sans ça, la position serait
        // structurellement identifiable comme un puzzle par le réseau, et
        // l'apprentissage tactique ne se transférerait pas en partie.
        bool replay_ok = true;
        for (const std::string& move : puzzle.moves) {
            if (!m_boards[game_idx].movePieceUCI(move)) {
                replay_ok = false;
                break;
            }
            m_stats.replayed_plies++;
        }

        if (replay_ok) {
            m_tactical_boost[game_idx] = true;
            m_replayed_plies[game_idx] = static_cast<int>(
                m_boards[game_idx].getMoveHistory().size());
        }
        else {
            // Rejeu impossible : on retombe sur une partie normale plutôt que
            // de partir d'une position corrompue.
            m_boards[game_idx].clear();
            m_boards[game_idx].setStartupPieces();
            m_tactical_boost[game_idx] = false;
        }
    }
    else {
        m_boards[game_idx].setStartupPieces();
        m_tactical_boost[game_idx] = false;
    }

    m_roots[game_idx] = std::make_unique<MCTSNode>(0.0f);
    m_sims_completed[game_idx] = 0;
    m_game_states[game_idx].clear();
    m_game_policies[game_idx].clear();

    expand_root(game_idx);

    if (m_tactical_boost[game_idx]) {
        // on force la tactique à trouver à être en slow move
        m_is_slow_move[game_idx] = true;
        m_sims_target[game_idx] = TACTICAL_FIRST_MOVE_SIMS;
        m_sims_completed[game_idx] = 0;
    }
    else {
        roll_next_move(game_idx);
    }

    // La decision centralisee lit le boost puzzle et la classification du coup
    // prepare par roll_next_move : le premier coup d'un puzzle garde 0,30, un
    // coup rapide peut retomber a zero selon le reglage.
    m_pending_epsilon[game_idx] = epsilon_for_move(game_idx);
    apply_pending_noise(game_idx);
}

void SelfPlayManager::apply_pending_noise(int game_idx) {
    if (m_pending_epsilon[game_idx] <= 0.0f) return;
    MCTSNode* root = m_roots[game_idx].get();
    if (root == nullptr || root->children.empty()) return;
    m_shared_mcts->add_dirichlet_noise(root, m_pending_epsilon[game_idx]);
    m_pending_epsilon[game_idx] = 0.0f;
}

void SelfPlayManager::execute_gpu_batch() {
    if (m_waiting_leaves.empty()) return;
    int current_batch_size = m_waiting_leaves.size();
    SelfPlayTiming* const timing = m_timing.enabled ? &m_timing : nullptr;
    if (timing != nullptr) {
        timing->batch_calls++;
        timing->batch_rows += static_cast<std::uint64_t>(current_batch_size);
        if (current_batch_size > static_cast<int>(timing->max_batch_rows)) {
            timing->max_batch_rows =
                static_cast<std::uint64_t>(current_batch_size);
        }
        timing->batch_histogram[selfplay_batch_bucket(
            static_cast<std::uint64_t>(current_batch_size))]++;
        if (m_last_batch_rows != static_cast<std::uint64_t>(current_batch_size)) {
            timing->batch_size_changes++;
        }
        m_last_batch_rows = static_cast<std::uint64_t>(current_batch_size);
    }

    auto cleanup_waiting = [&]() noexcept {
        for (int i = 0; i < current_batch_size; ++i) {
            const int game_idx = m_waiting_game_indices[i];
            for (int move = 0; move < m_waiting_moves_played[i]; ++move) {
                m_boards[game_idx].undoMove();
            }
            m_waiting_moves_played[i] = 0;
            m_is_waiting[game_idx] = false;
        }
        m_waiting_reservations.clear();
        m_waiting_leaves.clear();
        m_waiting_game_indices.clear();
        m_waiting_moves_played.clear();
    };

    try {
        const EvaluatorTotals totals_before = timing != nullptr
            ? m_evaluator->diagnostic_totals()
            : EvaluatorTotals{};
        {
            SelfPlayPhaseTimer evaluator_timer(
                timing, SelfPlayPhase::BatchEvaluator);
            // La reserve de fusion garde la capacite maximale ; l'appel a
            // l'evaluateur ne voit que la taille utile exacte.
            m_batch_input.resize(
                static_cast<std::size_t>(current_batch_size) * 119 * 64);
            m_evaluator->evaluate_batch(
                m_batch_input, m_batch_policies, m_batch_values,
                current_batch_size);
        }
        if (timing != nullptr) {
            const EvaluatorTotals delta = delta_totals(
                totals_before, m_evaluator->diagnostic_totals());
            timing->onnx_run_ns += delta.timing.run_ns;
            timing->onnx_softmax_ns += delta.timing.softmax_ns;
        }
        {
            SelfPlayPhaseTimer validation_timer(
                timing, SelfPlayPhase::BatchValidation);
            validate_network_output(
                m_batch_policies, m_batch_values, current_batch_size);
        }

        {
            SelfPlayPhaseTimer consume_timer(
                timing, SelfPlayPhase::BatchConsume);
            for (int i = 0; i < current_batch_size; ++i) {
                int game_idx = m_waiting_game_indices[i];
                int moves_played = m_waiting_moves_played[i];

                float value = m_batch_values[i];
                const float* single_policy =
                    m_batch_policies.data() + (i * 4672);
                m_shared_mcts->expand_and_backup(
                    m_waiting_leaves[i], m_boards[game_idx], single_policy,
                    value, m_waiting_reservations[i]);

                for (int k = 0; k < moves_played; ++k) {
                    m_boards[game_idx].undoMove();
                }
                m_waiting_moves_played[i] = 0;
                m_sims_completed[game_idx]++;
                if (timing != nullptr) timing->completed_sims++;
            }
        }

        // Une racine dont les enfants viennent d'etre materialises, par
        // expansion ou par hit de table pendant la descente, recoit ici le
        // bruit de Dirichlet en attente. Une seule application par coup.
        {
            SelfPlayPhaseTimer finalize_timer(
                timing, SelfPlayPhase::BatchFinalize);
            for (int i = 0; i < current_batch_size; ++i) {
                apply_pending_noise(m_waiting_game_indices[i]);
            }
        }
    }
    catch (...) {
        {
            SelfPlayPhaseTimer finalize_timer(
                timing, SelfPlayPhase::BatchFinalize);
            cleanup_waiting();
        }
        throw;
    }

    {
        SelfPlayPhaseTimer finalize_timer(
            timing, SelfPlayPhase::BatchFinalize);
        cleanup_waiting();
    }
}

void SelfPlayManager::play_best_move(int game_idx) {
    // 1. Calcul des probabilités de visite
    std::vector<float> pi(4672, 0.0f);
    float sum_visits = 0.0f;
    for (const auto& pair : m_roots[game_idx]->children) {
        pi[pair.first] = pair.second->visit_count;
        sum_visits += pair.second->visit_count;
    }
    if (sum_visits > 0.0f) {
        for (float& p : pi) p /= sum_visits;
    }

    // 2. Sélection
    int best_move = -1;

    // Sélection proportionnelle (30 premiers demi-coups)
    if (m_boards[game_idx].getMoveHistory().size() < 30) {
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float r = dis(m_rng);
        float accum = 0.0f;

        for (const auto& pair : m_roots[game_idx]->children) {
            float p = pi[pair.first];
            if (p > 0.0f) {
                accum += p;
                if (r <= accum) {
                    best_move = pair.first;
                    break;
                }
            }
        }
        if (best_move == -1) best_move = m_roots[game_idx]->children.front().first;
    }
    else { // argmax
        float max_p = -1.0f;
        for (const auto& pair : m_roots[game_idx]->children) {
            float p = pi[pair.first];
            if (p > max_p) { max_p = p; best_move = pair.first; }
        }
    }

    // 3. Sauvegarde (slow moves uniquement)
    if (m_is_slow_move[game_idx]) {
        std::vector<float> tensor;
        m_boards[game_idx].getAlphaZeroTensor(tensor);
        m_game_states[game_idx].push_back(std::move(tensor));
        m_game_policies[game_idx].push_back(std::move(pi));
        if (m_timing.enabled) m_timing.slow_examples_saved++;
    }

    // 4. Jouer le coup
    if (m_shared_mcts->apply_move_by_index(m_boards[game_idx], best_move)) {
        m_stats.new_plies++;
    }

    // 5. Détection fin de partie
    bool game_over = false;
    if (m_forced_end_plies[game_idx] > 0
        && static_cast<int>(m_boards[game_idx].getMoveHistory().size())
               >= m_forced_end_plies[game_idx]) {
        // Levier de test : fin imposee a un nombre de plies connu, sans
        // terminal reel ni attente. Toujours inactif en production.
        game_over = true;
    }
    else if (m_boards[game_idx].checkThreefoldRepetition() ||
        m_boards[game_idx].getHalfMoveClock() >= 100 ||
        m_boards[game_idx].checkInsufficientMaterial()) {
        game_over = true;
    }
    else if (!m_boards[game_idx].hasAnyLegalMove()) {
        game_over = true;
    }

    if (game_over) {
        m_roots[game_idx].reset();
        m_sims_target[game_idx] = 0;
        m_sims_completed[game_idx] = 0;
        return;
    }

    // 6. Descente de racine
    if (m_roots[game_idx]->has_child(best_move)) {
        m_roots[game_idx] = m_roots[game_idx]->extract_child(best_move);
        m_roots[game_idx]->parent = nullptr;
    }
    else {
        m_roots[game_idx] = std::make_unique<MCTSNode>(0.0f);
        expand_root(game_idx);
    }

    // Le renfort tactique ne vaut que pour le premier coup, celui où il y a une
    // tactique à trouver. Il est consomme apres que ce coup a ete joue et avant
    // de preparer le suivant : la classification slow/fast et le nouvel
    // epsilon ne le voient donc plus. Sans cette ligne, une partie amorcée par
    // un puzzle gardait 0,30 de bruit du début à la fin, et ses coups suivants
    // étaient enregistrés comme échantillons avec des cibles issues d'un jeu
    // plus bruité que la normale.
    m_tactical_boost[game_idx] = false;

    roll_next_move(game_idx);

    // Le bruit du nouveau coup est du a la racine, une seule fois. Si ses
    // enfants ne sont pas encore materialises, il reste en attente et sera
    // applique des qu'ils existent, dans execute_gpu_batch. Zero ecrase une
    // eventuelle valeur ancienne et n'appelle jamais add_dirichlet_noise.
    m_pending_epsilon[game_idx] = epsilon_for_move(game_idx);
    apply_pending_noise(game_idx);
}

void SelfPlayManager::roll_next_move(int game_idx) {
    // logique slow/fast moves pour accélérer la production de games.
    // en général : 1/4 des coups sont slow
    // quand 6 pièces ou moins : slow move + souvent
    // ça permet d'accélérer la compréhension des finales
    // (training supervisé sur des parties de GM :
    // peu d'exemples de mats)

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float random_val = dis(m_rng);

    int piece_count = m_boards[game_idx].getNumberOfOccupiedSquares();

    float effective_slow_ratio = (piece_count <= 6) ? 
        std::max(m_slow_ratio, 0.60f) : // 60% (au moins) si finale
        m_slow_ratio; // ratio normal sinon

    m_is_slow_move[game_idx] = (random_val < effective_slow_ratio);

    m_sims_target[game_idx] = m_is_slow_move[game_idx] ? m_slow_sims : m_fast_sims;
    m_sims_completed[game_idx] = 0;

    // Augmentation de données : on retire parfois l'historique pour que le
    // réseau sache fonctionner sans lui.
    //
    // Ce n'est PLUS un correctif de confondant : depuis que les positions de
    // puzzles portent l'historique réel de leur partie d'origine, l'absence
    // d'historique ne corrèle plus avec la tacticité. Le motif restant est la
    // robustesse aux entrées sans historique, cas de la FEN collée à la main
    // dans une GUI.
    //
    // Taux non mesuré. Le banc de puzzles pourra le valider en évaluant les
    // mêmes positions avec et sans historique.
    if (dis(m_rng) < 0.01f) {
        m_boards[game_idx].setAmnesiaMode(true);
    }
    else {
        m_boards[game_idx].setAmnesiaMode(false);
    }
}

std::vector<GameResult> SelfPlayManager::generate_games(int total_games_to_play) {
    if (total_games_to_play < 0) {
        throw std::invalid_argument(
            "generate_games : le quota doit etre positif ou nul");
    }

    // Une generation independante : aucune partie active heritee de l'appel
    // precedent, compteurs remis a zero.
    m_finished_games.clear();
    m_stats = SelfPlayStats{};
    for (int i = 0; i < m_num_concurrent_games; ++i) {
        m_roots[i].reset();
        m_slot_active[i] = false;
        m_is_waiting[i] = false;
        m_sims_completed[i] = 0;
        m_sims_target[i] = 0;
        m_pending_epsilon[i] = 0.0f;
        m_tactical_boost[i] = false;
    }

    // Diagnostic de debit : remis a zero a chaque generation, meme desactive,
    // pour qu'aucun appel n'herite du rapport precedent.
    m_timing = SelfPlayTiming{};
    m_timing.enabled = m_diagnostics_mode > 0;
    m_timing.mode = m_diagnostics_mode;
    m_last_batch_rows = 0;
    SelfPlayTiming* const timing = m_timing.enabled ? &m_timing : nullptr;
    m_evaluator->set_timing_enabled(m_timing.enabled);
    const SearchCounters counters_before = m_shared_mcts->get_counters();
    const SelfPlayTiming::Clock::time_point generation_start =
        SelfPlayTiming::Clock::now();

    // Debut de la vidange : note une seule fois, au depart de la derniere
    // partie du quota, avec les compteurs de lots a cet instant.
    bool vidange_notee = false;
    std::uint64_t debut_vidange_ns = 0;
    std::uint64_t appels_avant_vidange = 0;
    std::uint64_t lignes_avant_vidange = 0;
    auto noter_debut_vidange = [&]() {
        if (timing == nullptr || vidange_notee
            || m_stats.games_started
                   < static_cast<std::uint64_t>(total_games_to_play)) {
            return;
        }
        vidange_notee = true;
        debut_vidange_ns = elapsed_ns_since(generation_start);
        appels_avant_vidange = timing->batch_calls;
        lignes_avant_vidange = timing->batch_rows;
    };

    // Le quota est connu ici : au plus min(places, quota) parties demarrent.
    const int a_demarrer = std::min(m_num_concurrent_games,
                                    total_games_to_play);
    {
        SelfPlayPhaseTimer timer(timing, SelfPlayPhase::InitialSlots);
        for (int i = 0; i < a_demarrer; ++i) {
            demarrer_slot(i);
        }
    }
    noter_debut_vidange();

    int games_completed = 0;
    auto start_time = std::chrono::steady_clock::now();
    std::uint64_t pending_since_ns = 0;

    // Initialisation OpenMP
    const int num_threads = std::min(8, m_num_concurrent_games);
    std::vector<ThreadLocalBuffer> thread_buffers(num_threads);
    std::vector<SearchWorkCounters> worker_search_work(
        timing != nullptr ? static_cast<std::size_t>(num_threads) : 0);
    std::vector<std::uint64_t> worker_iter_sum(
        static_cast<std::size_t>(num_threads), 0);
    std::vector<std::uint64_t> worker_iter_max(
        static_cast<std::size_t>(num_threads), 0);
    // Compteurs ecrits par un seul worker chacun, agreges apres la barriere :
    // un increment direct depuis la region OpenMP perdrait des mises a jour.
    std::vector<std::uint64_t> worker_leaf_requests(
        static_cast<std::size_t>(num_threads), 0);
    std::vector<std::uint64_t> worker_no_network(
        static_cast<std::size_t>(num_threads), 0);

    for (auto& buf : thread_buffers) {
        buf.tensors.resize(m_num_concurrent_games * 119 * 64);
        buf.tensor_scratch.resize(119 * 64);
    }

    while (games_completed < total_games_to_play) {
        const SelfPlayTiming::Clock::time_point turn_start =
            SelfPlayTiming::Clock::now();
        if (timing != nullptr) ++timing->loop_turns;

        // ==========================================================
        // PHASE 1 : Séquentiel - Jouer les coups, gérer les fins
        // ==========================================================
        {
            SelfPlayPhaseTimer phase_timer(
                timing, SelfPlayPhase::MoveManagement);
            for (int i = 0; i < m_num_concurrent_games; ++i) {
                if (games_completed >= total_games_to_play) break;
                if (!m_slot_active[i] || m_is_waiting[i]) continue;

                if (m_sims_completed[i] >= m_sims_target[i] && m_sims_target[i] > 0) {
                    play_best_move(i);

                    if (m_roots[i] != nullptr && longueur_max_atteinte(i)) {
                        m_roots[i].reset();
                        m_sims_target[i] = 0;
                        m_sims_completed[i] = 0;
                    }

                    if (m_roots[i] == nullptr) {
                        GameResult res;
                        res.move_count = m_game_states[i].size();
                        res.total_real_moves = m_boards[i].getMoveHistory().size() + m_boards[i].getInitialPlyOffset();
                        res.flat_states.reserve(res.move_count * 119 * 64);
                        res.flat_policies.reserve(res.move_count * 4672);

                        for (const auto& t : m_game_states[i])
                            res.flat_states.insert(res.flat_states.end(), t.begin(), t.end());
                        for (const auto& p : m_game_policies[i])
                            res.flat_policies.insert(res.flat_policies.end(), p.begin(), p.end());

                        // Les donnees par coup appartiennent desormais au
                        // resultat : la place libere sa copie au lieu de la
                        // garder jusqu'a un eventuel redemarrage.
                        m_game_states[i].clear();
                        m_game_policies[i].clear();

                        const GameConclusion conclusion = conclure_partie(
                            m_boards[i], longueur_max_atteinte(i));
                        res.final_outcome = conclusion.final_outcome;
                        res.end_reason = conclusion.end_reason;

                        m_finished_games.push_back(std::move(res));
                        games_completed++;
                        m_stats.games_completed++;

                        if (games_completed % 16 == 0 || games_completed == total_games_to_play) {
                            auto now = std::chrono::steady_clock::now();
                            double elapsed = std::chrono::duration<double>(now - start_time).count();
                            double speed = games_completed / elapsed;
                            double eta = (total_games_to_play - games_completed) / speed;
                            std::cout << "\r  Self-play: " << games_completed << "/" << total_games_to_play
                                << " (" << std::fixed << std::setprecision(1) << speed << " parties/s"
                                << ", ETA: " << (int)(eta / 60) << "m" << (int)((int)eta % 60) << "s)"
                                << std::flush;
                        }

                        if (m_stats.games_started
                                < static_cast<std::uint64_t>(
                                      total_games_to_play)) {
                            demarrer_slot(i);
                            noter_debut_vidange();
                        }
                        else {
                            // Plus aucun depart a effectuer : la place reste
                            // inactive et les parties engagees finissent.
                            m_slot_active[i] = false;
                        }
                    }
                }
            }
        }

        // ==========================================================
        // PHASE 2 : Parallèle (OpenMP) - Traversée MCTS
        // ==========================================================
        for (auto& buf : thread_buffers) buf.clear();
        const bool detail_workers = (timing != nullptr && timing->mode >= 2);
        if (detail_workers) {
            timing->worker_count = static_cast<std::uint64_t>(num_threads);
        }

        {
            SelfPlayPhaseTimer collection_timer(
                timing, SelfPlayPhase::Collection);
#pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                auto& buf = thread_buffers[tid];
                SearchWorkCounters* const work = timing != nullptr
                    ? &worker_search_work[static_cast<std::size_t>(tid)]
                    : nullptr;
                std::uint64_t iter_sum = 0;
                std::uint64_t iter_max = 0;
                std::uint64_t requests_thread = 0;
                std::uint64_t no_network_thread = 0;

#pragma omp for schedule(dynamic, 4)
                for (int i = 0; i < m_num_concurrent_games; ++i) {
                    const SelfPlayTiming::Clock::time_point item_start =
                        detail_workers
                            ? SelfPlayTiming::Clock::now()
                            : SelfPlayTiming::Clock::time_point{};
                    [&]() {
                        if (!m_slot_active[i] || m_is_waiting[i]
                            || m_sims_completed[i] >= m_sims_target[i]) {
                            return;
                        }

                        int moves_played = 0;
                        PathReservation reservation;
                        MCTSNode* leaf = m_shared_mcts->advance_to_leaf(
                            m_roots[i].get(), m_boards[i], 1.4f, moves_played,
                            reservation, work);

                        if (leaf != nullptr) {
                            requests_thread++;
                            buf.leaves.push_back(leaf);
                            buf.game_indices.push_back(i);
                            buf.moves_played.push_back(moves_played);
                            buf.reservations.push_back(std::move(reservation));

                            m_boards[i].getAlphaZeroTensor(buf.tensor_scratch);
                            int offset = (buf.leaves.size() - 1) * 119 * 64;
                            std::copy(buf.tensor_scratch.begin(),
                                buf.tensor_scratch.end(),
                                buf.tensors.begin() + offset);

                            m_is_waiting[i] = true;
                        }
                        else {
                            m_sims_completed[i]++;
                            no_network_thread++;
                        }
                    }();
                    if (detail_workers) {
                        const std::uint64_t item_ns =
                            elapsed_ns_since(item_start);
                        iter_sum += item_ns;
                        if (item_ns > iter_max) iter_max = item_ns;
                    }
                }

                worker_leaf_requests[static_cast<std::size_t>(tid)] =
                    requests_thread;
                worker_no_network[static_cast<std::size_t>(tid)] =
                    no_network_thread;
                if (detail_workers) {
                    worker_iter_sum[static_cast<std::size_t>(tid)] = iter_sum;
                    worker_iter_max[static_cast<std::size_t>(tid)] = iter_max;
                }
            }
        }
        if (timing != nullptr) {
            std::uint64_t no_network_turn = 0;
            for (std::size_t worker = 0; worker < worker_leaf_requests.size();
                 ++worker) {
                timing->leaf_requests += worker_leaf_requests[worker];
                no_network_turn += worker_no_network[worker];
            }
            timing->no_network_sims += no_network_turn;
            timing->completed_sims += no_network_turn;
        }
        if (detail_workers) {
            for (std::size_t worker = 0; worker < worker_iter_sum.size();
                 ++worker) {
                timing->worker_busy_sum_ns += worker_iter_sum[worker];
                if (worker_iter_max[worker] > timing->worker_busy_max_ns) {
                    timing->worker_busy_max_ns = worker_iter_max[worker];
                }
            }
        }

        // ==========================================================
        // PHASE 3 : Séquentiel - Fusion et batch GPU
        // ==========================================================
        bool batch_full = false;
        bool all_blocked = true;
        {
            SelfPlayPhaseTimer assembly_timer(
                timing, SelfPlayPhase::AssemblyDispatch);
            // Capacite maximale avant fusion : la reduction a la taille utile
            // appartient a la phase d'evaluation, qui la refait juste avant
            // l'appel ONNX.
            m_batch_input.resize(
                static_cast<std::size_t>(m_num_concurrent_games) * 119 * 64);
            for (auto& buf : thread_buffers) {
                for (size_t j = 0; j < buf.leaves.size(); ++j) {
                    if ((int)m_waiting_leaves.size() >= m_num_concurrent_games) break;

                    m_waiting_leaves.push_back(buf.leaves[j]);
                    m_waiting_game_indices.push_back(buf.game_indices[j]);
                    m_waiting_moves_played.push_back(buf.moves_played[j]);
                    m_waiting_reservations.push_back(
                        std::move(buf.reservations[j]));

                    int batch_offset = m_waiting_leaves.size() - 1;
                    std::copy(buf.tensors.begin() + j * 119 * 64,
                        buf.tensors.begin() + (j + 1) * 119 * 64,
                        m_batch_input.begin() + batch_offset * 119 * 64);
                }
            }

            // Exécution : batch plein OU tout le monde est bloqué.
            //
            // Le report est volontaire : une place qui enchaine les
            // simulations sans reseau peut retarder le lot, mais lancer a
            // chaque tour casse la stabilite des tailles de lot. Mesure du
            // 2026-09-25 : 30 474 changements de forme contre 238, et ONNX
            // Runtime repayait son plan a chaque appel, 2,6 fois plus lent.
            batch_full = ((int)m_waiting_leaves.size() >= m_num_concurrent_games);

            for (int i = 0; i < m_num_concurrent_games; ++i) {
                if (m_slot_active[i] && m_sims_completed[i] < m_sims_target[i]
                    && !m_is_waiting[i]) {
                    all_blocked = false;
                    break;
                }
            }
        }

        if ((batch_full || all_blocked) && !m_waiting_leaves.empty()) {
            execute_gpu_batch();
            if (timing != nullptr && pending_since_ns != 0) {
                const std::uint64_t age_ns =
                    elapsed_ns_since(generation_start) - pending_since_ns;
                if (age_ns > timing->max_pending_age_ns) {
                    timing->max_pending_age_ns = age_ns;
                }
                pending_since_ns = 0;
            }
        }
        else if (timing != nullptr && !m_waiting_leaves.empty()) {
            timing->deferred_turns++;
            timing->deferred_wall_ns += elapsed_ns_since(turn_start);
            if (pending_since_ns == 0) {
                pending_since_ns = elapsed_ns_since(generation_start);
            }
        }
    }

    if (timing != nullptr) {
        for (const SearchWorkCounters& work : worker_search_work) {
            timing->search_descents += work.descents;
            timing->search_depth_sum += work.depth_sum;
            timing->search_depth_max = std::max(
                timing->search_depth_max, work.depth_max);
            timing->search_children_scored += work.children_scored;
        }
        timing->generation_wall_ns = elapsed_ns_since(generation_start);
        if (vidange_notee) {
            timing->drain_wall_ns = timing->generation_wall_ns - debut_vidange_ns;
            timing->drain_batch_calls = timing->batch_calls - appels_avant_vidange;
            timing->drain_batch_rows = timing->batch_rows - lignes_avant_vidange;
        }
        std::uint64_t phases_ns = 0;
        for (const std::uint64_t phase_ns : timing->phase_wall_ns) {
            phases_ns += phase_ns;
        }
        timing->generation_other_wall_ns =
            static_cast<std::int64_t>(timing->generation_wall_ns)
            - static_cast<std::int64_t>(phases_ns);

        const SearchCounters counters_after = m_shared_mcts->get_counters();
        timing->terminal_sims =
            counters_after.terminal_hits - counters_before.terminal_hits;
        timing->tt_hits = counters_after.tt_hits - counters_before.tt_hits;
        timing->tt_misses =
            counters_after.tt_misses - counters_before.tt_misses;
    }

    std::cout << std::endl;
    // Les donnees appartiennent a l'appelant : le membre est vide apres le
    // retour, sans copie du vecteur ni des GameResult.
    return std::move(m_finished_games);
}

void SelfPlayManager::load_tactical_puzzles(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string line;
    int malformed = 0;

    // Un chemin relatif depend du repertoire de lancement : une erreur ici
    // desactive silencieusement l'injection de puzzles pendant toute la
    // campagne, donc on la signale fort.
    if (!file.is_open()) {
        std::cerr << "ATTENTION : puzzles tactiques illisibles ("
                  << filepath << "), injection desactivee." << std::endl;
        return;
    }

    // Format : <fen_initiale>|<coups_uci>|<solution>|<rating>|<themes>
    // Seuls les deux premiers champs nous concernent.
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        const size_t first = line.find('|');
        if (first == std::string::npos) { malformed++; continue; }
        const size_t second = line.find('|', first + 1);

        TacticalPuzzle puzzle;
        puzzle.start_fen = line.substr(0, first);

        const std::string moves_field = (second == std::string::npos)
            ? line.substr(first + 1)
            : line.substr(first + 1, second - first - 1);

        std::istringstream iss(moves_field);
        std::string move;
        while (iss >> move) puzzle.moves.push_back(move);

        if (puzzle.start_fen.empty()) { malformed++; continue; }
        m_tactical_puzzles.push_back(std::move(puzzle));
    }

    std::cout << "Charge " << m_tactical_puzzles.size()
              << " puzzles tactiques avec historique." << std::endl;
    if (malformed > 0) {
        std::cout << "  " << malformed << " ligne(s) mal formee(s) ignoree(s)."
                  << std::endl;
    }
}

void SelfPlayManager::load_start_positions(const std::string& filepath) {
    // Un banc qui retomberait en silence sur la position initiale mesurerait
    // autre chose que ce qu'il annonce : un fichier illisible ou vide leve.
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error(
            "positions de depart illisibles : " + filepath);
    }

    std::string line;
    int ecartees = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        Chessboard essai;
        essai.loadFEN(line);
        if (!essai.hasAnyLegalMove()) { ecartees++; continue; }
        m_start_fens.push_back(line);
    }

    if (m_start_fens.empty()) {
        throw std::runtime_error(
            "aucune position de depart jouable dans " + filepath);
    }
    std::cout << "Charge " << m_start_fens.size() << " positions de depart";
    if (ecartees > 0) std::cout << " (" << ecartees << " sans coup legal ecartees)";
    std::cout << "." << std::endl;
}
