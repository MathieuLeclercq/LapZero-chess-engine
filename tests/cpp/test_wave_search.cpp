#include "controlled_evaluator.hpp"
#include "discriminating_evaluator.hpp"
#include "mcts.hpp"
#include "mcts_test_access.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

class EvaluationGate {
public:
    void block() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_blocked = true;
        m_cv.notify_all();
        m_cv.wait(lock, [&]() { return m_released; });
    }

    void wait_until_blocked() {
        std::unique_lock<std::mutex> lock(m_mutex);
        require_test(m_cv.wait_for(lock, 2s, [&]() { return m_blocked; }),
                     "evaluator did not block");
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_released = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_blocked = false;
    bool m_released = false;
};

Chessboard startup_board() {
    Chessboard board;
    board.setStartupPieces();
    return board;
}

void require_quiescent(const TreeReport& report) {
    require_test(report.en_vol == 0, "tree retained in-flight nodes");
    require_test(report.pending == 0, "tree retained Pending nodes");
    require_test(report.violations == 0, "tree invariant failed");
}

void require_best_move_is_legal(MCTS& mcts, Chessboard& board) {
    const std::vector<MoveStats> stats = mcts.get_analysis_results();
    if (stats.empty()) return;
    const std::vector<int> legal = board.getLegalMoveIndices();
    require_test(std::find(legal.begin(), legal.end(), stats.front().move_idx)
                     != legal.end(),
                 "best move is illegal on root board");
}

Chessboard rejouer_chemin(const Chessboard& depart, MCTS& mcts,
                          const std::vector<int>& chemin) {
    Chessboard replay = depart;
    for (int idx : chemin) {
        require_test(mcts.apply_move_by_index(replay, idx),
                     "could not replay a tree path");
    }
    return replay;
}

// Verifie, pour chaque noeud materialise dont le tenseur a ete servi a
// l'evaluateur, que la value et les priors viennent de SA ligne. Un decalage
// entre lignes et feuilles, ou une ligne de padding consommee, echoue ici. Un
// tenseur non servi (transposition d'historique sur un hit TT) est ignore.
void verifier_association(MCTS& mcts, const Chessboard& depart,
                          MCTSNode* node, std::vector<int>& chemin,
                          const DiscriminatingEvaluator& evaluator) {
    Chessboard plateau = rejouer_chemin(depart, mcts, chemin);
    std::vector<float> tensor;
    plateau.getAlphaZeroTensor(tensor);

    if (evaluator.a_une_reponse(tensor)
        && node->state.load() == NodeState::Expanded) {
        const auto& reponse = evaluator.reponse_pour(tensor);
        require_test(std::fabs(node->network_value - reponse.value) < 1e-6f,
                     "a node carries the value of another batch row");
        const std::vector<int> legal = plateau.getLegalMoveIndices();
        float somme = 0.0f;
        for (int idx : legal) somme += reponse.policy[static_cast<std::size_t>(idx)];
        require_test(somme > 0.0f, "fixture policy has no legal mass");
        require_test(node->children.size() == legal.size(),
                     "expanded node does not expose all legal children");
        for (const auto& child : node->children) {
            const float attendu =
                reponse.policy[static_cast<std::size_t>(child.first)] / somme;
            require_test(std::fabs(child.second->prior - attendu) < 1e-5f,
                         "a child carries the prior of another batch row");
        }
    }

    for (const auto& child : node->children) {
        chemin.push_back(child.first);
        verifier_association(mcts, depart, child.second.get(), chemin,
                             evaluator);
        chemin.pop_back();
    }
}

void test_budgets_and_invariants_across_configurations() {
    constexpr std::array<int, 3> WORKERS{2, 4, 8};
    constexpr std::array<int, 2> BATCHES{1, 8};
    constexpr std::array<int, 6> BUDGETS{0, 1, 3, 8, 17, 100};

    for (int workers : WORKERS) {
        for (int batch : BATCHES) {
            ControlledEvaluator evaluator;
            MCTS mcts(&evaluator, 8192, 0);
            Chessboard board = startup_board();
            const std::string fen_before = board.toFEN();

            for (int budget : BUDGETS) {
                const TreeReport before = mcts.inspect_tree();
                const SearchCounters counters_before = mcts.get_counters();
                mcts.step_analysis(
                    board, budget, 1.4f, batch, workers);
                const TreeReport after = mcts.inspect_tree();
                const SearchCounters counters_after = mcts.get_counters();

                require_test(
                    after.root_visits - before.root_visits
                        == static_cast<std::uint64_t>(budget),
                    "multicore search completed the wrong visit budget");
                require_test(
                    counters_after.completed_simulations
                        - counters_before.completed_simulations
                        == static_cast<std::uint64_t>(budget),
                    "completed_simulations does not match budget");
                require_quiescent(after);
                require_test(board.toFEN() == fen_before,
                             "multicore search changed input board");
                if (budget > 0) require_best_move_is_legal(mcts, board);
            }
        }
    }
}

void test_terminal_root_uses_root_visits() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.loadFEN(
        "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");

    mcts.step_analysis(board, 17, 1.4f, 8, 4);
    const TreeReport report = mcts.inspect_tree();
    const SearchCounters counters = mcts.get_counters();

    require_test(report.root_visits == 17,
                 "terminal root lost completed simulations");
    require_test(report.nodes == 1, "terminal root grew children");
    require_quiescent(report);
    require_test(counters.terminal_hits == 17,
                 "known terminal hits were not counted");
    require_test(counters.completed_simulations == 17,
                 "terminal simulations were not completed");
}

void test_evaluator_failure_cleans_session_and_allows_recovery() {
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 2;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();

    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 4);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "controlled evaluator failure was swallowed");
    require_quiescent(mcts.inspect_tree());

    evaluator.fail_on_call = 0;
    mcts.step_analysis(board, 3, 1.4f, 3, 2);
    const TreeReport recovered = mcts.inspect_tree();
    require_quiescent(recovered);
    require_test(recovered.root_visits == 3,
                 "search did not recover after evaluator failure");
}

void test_invalid_evaluator_outputs_clean_session() {
    for (bool truncate_policy : {true, false}) {
        ControlledEvaluator evaluator;
        if (truncate_policy) evaluator.truncate_policy_on_call = 2;
        else evaluator.truncate_values_on_call = 2;
        MCTS mcts(&evaluator, 8192, 0);
        Chessboard board = startup_board();

        bool failed = false;
        try {
            mcts.step_analysis(board, 8, 1.4f, 8, 4);
        }
        catch (const std::runtime_error&) {
            failed = true;
        }
        require_test(failed, "invalid evaluator dimensions were accepted");
        require_quiescent(mcts.inspect_tree());
    }

    ControlledEvaluator evaluator;
    evaluator.output_value = std::numeric_limits<float>::quiet_NaN();
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 4);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "non-finite evaluator value was accepted");
    require_quiescent(mcts.inspect_tree());
}

void test_fixed_batch_pads_wave_calls_and_keeps_budget() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    require_test(!mcts.fixed_batch(),
                 "le lot fixe doit etre desactive par defaut");
    mcts.set_fixed_batch(true);
    require_test(mcts.fixed_batch(), "le lot fixe n'a pas ete active");

    Chessboard board = startup_board();
    mcts.step_analysis(board, 17, 1.4f, 8, 4);

    require_test(!evaluator.batch_sizes.empty(), "evaluateur jamais appele");
    require_test(evaluator.batch_sizes.front() == 1,
                 "l'expansion de racine doit rester a batch 1");
    for (std::size_t i = 1; i < evaluator.batch_sizes.size(); ++i) {
        require_test(evaluator.batch_sizes[i] == 8,
                     "un appel de vague n'a pas la forme fixe");
    }

    const TreeReport report = mcts.inspect_tree();
    const SearchCounters counters = mcts.get_counters();
    require_test(report.root_visits == 17, "budget de vagues incorrect");
    require_test(counters.completed_simulations == 17,
                 "simulations terminees incorrectes");
    require_quiescent(report);
    require_test(board.toFEN() == startup_board().toFEN(),
                 "le lot fixe a modifie le plateau d'entree");
}

void test_fixed_batch_pads_mono_batch() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_fixed_batch(true);
    Chessboard board = startup_board();
    mcts.mcts_search(board, 17, 1.4f, false, 8, 1);

    require_test(!evaluator.batch_sizes.empty(), "evaluateur jamais appele");
    require_test(evaluator.batch_sizes.front() == 1,
                 "l'expansion de racine doit rester a batch 1");
    for (std::size_t i = 1; i < evaluator.batch_sizes.size(); ++i) {
        require_test(evaluator.batch_sizes[i] == 8,
                     "un lot mono n'a pas la forme fixe");
    }
}

void test_tuning_defaults_validation_and_propagation() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);

    const SearchTuning defauts = mcts.get_tuning();
    require_test(defauts.virtual_loss == 1
                     && defauts.fpu_reduction == 0.30f
                     && defauts.collision_attempt_factor == 4,
                 "les reglages de divergence par defaut ont change");

    mcts.set_tuning(SearchTuning{2, 0.5f, 8});
    const SearchTuning modifies = mcts.get_tuning();
    require_test(modifies.virtual_loss == 2
                     && modifies.fpu_reduction == 0.5f
                     && modifies.collision_attempt_factor == 8,
                 "set_tuning n'a pas propage les valeurs");

    bool refuse = false;
    try {
        mcts.set_tuning(SearchTuning{0, 0.3f, 4});
    }
    catch (const std::invalid_argument&) {
        refuse = true;
    }
    require_test(refuse, "virtual_loss nul accepte");
}

void test_each_leaf_gets_the_value_and_priors_of_its_own_tensor() {
    DiscriminatingEvaluator evaluator;
    Chessboard board = startup_board();
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_fixed_batch(true);
    mcts.step_analysis(board, 24, 1.4f, 8, 4);

    require_quiescent(mcts.inspect_tree());
    std::vector<int> chemin;
    verifier_association(mcts, board, MCTSTestAccess::analysis_root(mcts),
                         chemin, evaluator);
}

void test_terminal_leaves_do_not_shift_network_rows() {
    DiscriminatingEvaluator evaluator;
    Chessboard board = startup_board();
    const std::vector<int> legal = board.getLegalMoveIndices();
    MCTS mcts(&evaluator, 8192, 0);

    // Racine fabriquee : trois enfants, dont le premier est deja terminal.
    MCTSNode root(0.0f);
    root.state.store(NodeState::Expanded);
    root.visit_count = 8;  // exploration_factor non nul : le virtual loss ecarte
    const float prior = 1.0f / 3.0f;
    for (int i = 0; i < 3; ++i) {
        const int move = legal[static_cast<std::size_t>(i)];
        root.children.emplace_back(
            move, std::make_unique<MCTSNode>(prior, move, &root));
    }
    root.children[0].second->state.store(NodeState::Terminal);

    MCTSTestAccess::run_waves(mcts, &root, board, 3, 1.4f, 8, 4);

    require_test(evaluator.batch_sizes == std::vector<int>{2},
                 "terminal leaves consumed or shifted a batch row");
    require_test(mcts.get_counters().nn_calls == 2,
                 "nn_calls does not match the network leaves");
    require_test(mcts.get_counters().completed_simulations == 3,
                 "the wave did not complete its three simulations");
    std::vector<int> chemin;
    verifier_association(mcts, board, &root, chemin, evaluator);
}

void test_partial_waves_pad_to_the_fixed_shape() {
    for (int budget : {1, 3, 7}) {
        DiscriminatingEvaluator evaluator;
        Chessboard board = startup_board();
        MCTS mcts(&evaluator, 8192, 0);
        mcts.set_fixed_batch(true);
        mcts.step_analysis(board, budget, 1.4f, 8, 4);

        require_quiescent(mcts.inspect_tree());
        require_test(!evaluator.batch_sizes.empty(), "no evaluation at all");
        require_test(evaluator.batch_sizes.front() == 1,
                     "root expansion must stay at batch one");
        for (std::size_t i = 1; i < evaluator.batch_sizes.size(); ++i) {
            require_test(evaluator.batch_sizes[i] == 8,
                         "a partial wave lost its fixed shape");
        }
        std::vector<int> chemin;
        verifier_association(mcts, board,
                             MCTSTestAccess::analysis_root(mcts), chemin,
                             evaluator);
    }

    DiscriminatingEvaluator evaluator;
    Chessboard board = startup_board();
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_fixed_batch(true);
    mcts.step_analysis(board, 3, 1.4f, 32, 4);
    require_test(evaluator.batch_sizes.size() >= 2
                     && evaluator.batch_sizes.front() == 1,
                 "root expansion must stay at batch one");
    for (std::size_t i = 1; i < evaluator.batch_sizes.size(); ++i) {
        require_test(evaluator.batch_sizes[i] == 32,
                     "batch 32 padding was not applied");
    }
    std::vector<int> chemin;
    verifier_association(mcts, board, MCTSTestAccess::analysis_root(mcts),
                         chemin, evaluator);
}

void test_terminal_root_never_sends_a_wave() {
    DiscriminatingEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.loadFEN(
        "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");

    mcts.step_analysis(board, 17, 1.4f, 8, 4);

    require_test(evaluator.batch_sizes.empty(),
                 "a terminal root sent an evaluation batch");
    require_quiescent(mcts.inspect_tree());
}

void test_padding_sentinels_are_ignored_and_counters_stay_honest() {
    DiscriminatingEvaluator evaluator;
    Chessboard board = startup_board();
    evaluator.sentinelles_padding = true;
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_fixed_batch(true);
    mcts.step_analysis(board, 17, 1.4f, 8, 4);

    require_quiescent(mcts.inspect_tree());
    std::vector<int> chemin;
    verifier_association(mcts, board, MCTSTestAccess::analysis_root(mcts),
                         chemin, evaluator);

    const SearchCounters counters = mcts.get_counters();
    require_test(
        counters.nn_batches
            == static_cast<std::uint64_t>(evaluator.batch_sizes.size()),
        "nn_batches does not count physical calls");
    std::uint64_t lignes = 0;
    for (int taille : evaluator.batch_sizes) {
        lignes += static_cast<std::uint64_t>(taille);
    }
    require_test(counters.nn_calls <= lignes,
                 "nn_calls exceeds the physical rows of the batches");
}

class PermutingEvaluator final : public Evaluator {
public:
    DiscriminatingEvaluator inner;

    void evaluate_batch(const std::vector<float>& input,
                        std::vector<float>& policies,
                        std::vector<float>& values,
                        int batch_size) override {
        inner.evaluate_batch(input, policies, values, batch_size);
        if (batch_size < 2) return;
        std::swap(values[0], values[1]);
        for (int k = 0; k < POLICY_SIZE; ++k) {
            std::swap(policies[static_cast<std::size_t>(k)],
                      policies[static_cast<std::size_t>(POLICY_SIZE + k)]);
        }
    }
};

void test_the_association_check_detects_a_permutation() {
    PermutingEvaluator evaluator;
    Chessboard board = startup_board();
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_fixed_batch(true);
    mcts.step_analysis(board, 17, 1.4f, 8, 4);

    bool detected = false;
    try {
        std::vector<int> chemin;
        verifier_association(mcts, board,
                             MCTSTestAccess::analysis_root(mcts), chemin,
                             evaluator.inner);
    }
    catch (const std::exception&) {
        detected = true;
    }
    require_test(detected, "permuted batch rows were not detected");
}

void test_mate_at_hundred_is_a_loss_in_every_search_path() {
    const char* mate_fen = "7k/6Q1/5K2/8/8/8/8/8 b - - 100 1";

    // Sequentiel (batch 0), mono batche (batch 8, 1 worker) et vagues.
    for (int batch : {0, 8}) {
        for (int workers : {1, 4}) {
            if (batch == 0 && workers != 1) continue;
            ControlledEvaluator evaluator;
            evaluator.fail_on_call = 1;  // aucune inference ne doit avoir lieu
            MCTS mcts(&evaluator, 8192, 0);
            Chessboard board;
            board.loadFEN(mate_fen);

            mcts.step_analysis(board, 8, 1.4f, batch, workers);

            require_test(evaluator.batch_sizes.empty(),
                         "a mated root called the evaluator");
            const TreeReport report = mcts.inspect_tree();
            require_quiescent(report);
            require_test(report.root_visits == 8,
                         "mated root lost simulations");
            require_test(report.nodes == 1,
                         "mated root grew children");
            require_test(std::fabs(mcts.get_root_q() + 1.0f) < 1e-6f,
                         "mate at the hundredth half-move was not a loss");
            require_test(mcts.get_counters().terminal_hits == 8,
                         "known terminal hits were not counted");
        }
    }
}

void test_scalar_search_restores_the_board_after_an_evaluator_error() {
    // Le deuxieme appel reseau echoue : la racine est deja developpee, donc
    // l'erreur tombe sur une feuille, en pleine descente.
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 2;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    const std::string avant = board.toFEN();

    bool leve = false;
    try {
        mcts.step_analysis(board, 4, 1.4f, 0, 1);
    }
    catch (const std::exception&) {
        leve = true;
    }
    require_test(leve, "the evaluator error did not propagate");
    require_test(board.toFEN() == avant,
                 "the scalar search left the board on the leaf");

    // La reprise sur le meme arbre doit aboutir, sans noeud reste reserve.
    evaluator.fail_on_call = 0;
    mcts.step_analysis(board, 8, 1.4f, 0, 1);
    require_test(mcts.inspect_tree().root_visits == 8,
                 "the search did not resume after the error");
    require_test(board.toFEN() == avant,
                 "the resumed search left the board on a leaf");
}

void test_mate_delivered_at_hundred_from_ninety_nine() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.loadFEN("7k/8/5KQ1/8/8/8/8/8 w - - 99 1");

    mcts.step_analysis(board, 200, 1.4f, 0, 1);

    MCTSNode* root = MCTSTestAccess::analysis_root(mcts);
    require_test(root != nullptr, "no analysis root");
    MCTSNode* mat = nullptr;
    for (const auto& child : root->children) {
        Chessboard apres = board;
        require_test(mcts.apply_move_by_index(apres, child.first),
                     "could not replay a root move");
        if (!apres.hasAnyLegalMove() && apres.isInCheck()) {
            require_test(mat == nullptr, "the fixture has several mating moves");
            mat = child.second.get();
        }
    }
    require_test(mat != nullptr, "the mating move was not materialised");
    require_test(mat->state.load() == NodeState::Terminal,
                 "the mating move was not classified terminal");
    require_test(std::fabs(mat->network_value + 1.0f) < 1e-6f,
                 "mate at the hundredth half-move was not a loss");
    require_test(mcts.get_root_q() > 0.0f,
                 "the root did not score the mate for White");
}

void test_terminal_classification_prefers_mate_over_rule_draws() {
    struct Cas {
        const char* fen;
        float attendu;
        const char* message;
    };
    const Cas cas[] = {
        {"7k/6Q1/5K2/8/8/8/8/8 b - - 100 1", -1.0f,
         "mate at the hundredth half-move"},
        {"7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", -1.0f, "mate before 100"},
        {"7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0.0f, "stalemate"},
        {"8/8/8/8/8/8/8/K6k w - - 0 1", 0.0f, "insufficient material"},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1", 0.0f,
         "true fifty-move draw"},
        {"8/8/8/8/8/8/4r3/4K2k w - - 100 1", 0.0f, "check with an escape"},
    };

    for (const Cas& c : cas) {
        ControlledEvaluator evaluator;
        evaluator.fail_on_call = 1;
        MCTS mcts(&evaluator, 8192, 0);
        Chessboard board;
        board.loadFEN(c.fen);
        MCTSNode root(0.0f);

        const float value = mcts.expand_node_single(&root, board);

        require_test(std::fabs(value - c.attendu) < 1e-6f, c.message);
        require_test(root.state.load() == NodeState::Terminal, c.message);
        require_test(evaluator.batch_sizes.empty(),
                     "a terminal position called the evaluator");
    }
}

void test_repetition_is_a_draw_in_the_search() {
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 1;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.setStartupPieces();
    const char* moves[] = {"g1f3", "g8f6", "f3g1", "f6g8",
                           "g1f3", "g8f6", "f3g1", "f6g8"};
    for (const char* uci : moves) {
        require_test(board.movePieceUCI(uci),
                     "a scripted repetition move failed");
    }

    MCTSNode root(0.0f);
    const float value = mcts.expand_node_single(&root, board);

    require_test(std::fabs(value) < 1e-6f, "repetition was not a draw");
    require_test(root.state.load() == NodeState::Terminal,
                 "repetition was not classified terminal");
    require_test(evaluator.batch_sizes.empty(),
                 "a repetition called the evaluator");
}

std::vector<float> politique_non_uniforme(const std::vector<int>& legal) {
    std::vector<float> policy(POLICY_SIZE, 0.0f);
    for (std::size_t i = 0; i < legal.size(); ++i) {
        policy[static_cast<std::size_t>(legal[i])] =
            1.0f + 0.05f * static_cast<float>(i);
    }
    return policy;
}

std::vector<float> priors_des_enfants(const MCTSNode& node) {
    std::vector<float> priors;
    for (const auto& child : node.children) {
        priors.push_back(child.second->prior);
    }
    return priors;
}

void test_tt_hit_expands_the_root_and_does_not_pollute_the_cache() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard board = startup_board();
    const std::vector<int> legal = board.getLegalMoveIndices();
    const std::vector<float> policy = politique_non_uniforme(legal);
    MCTSTestAccess::store(mcts, board, legal, policy, 0.25f);

    MCTSNode root(0.0f);
    const float value = mcts.expand_node_single(&root, board);

    require_test(evaluator.batch_sizes.empty(),
                 "a TT hit called the evaluator");
    require_test(value == 0.25f, "the TT value was not returned");
    require_test(root.state.load() == NodeState::Expanded,
                 "a TT hit left the root unexpanded");
    require_test(root.children.size() == legal.size(),
                 "a TT hit lost children");
    require_test(root.visit_count == 0 && root.total_value == 0.0f,
                 "the TT expansion performed a backup");
    const std::vector<float> priors_avant = priors_des_enfants(root);
    float somme = 0.0f;
    for (float prior : priors_avant) somme += prior;
    require_test(std::fabs(somme - 1.0f) < 1e-5f,
                 "cached priors do not sum to one");

    MCTSTestAccess::seed_noise(mcts, 1234);
    mcts.add_dirichlet_noise(&root, 0.5f);
    const std::vector<float> priors_bruites = priors_des_enfants(root);
    require_test(priors_bruites != priors_avant,
                 "noise did not change the root priors");

    MCTSNode second(0.0f);
    mcts.expand_node_single(&second, board);
    const std::vector<float> priors_second = priors_des_enfants(second);
    require_test(priors_second.size() == priors_avant.size(),
                 "the second root lost children");
    for (std::size_t i = 0; i < priors_avant.size(); ++i) {
        require_test(std::fabs(priors_second[i] - priors_avant[i]) < 1e-6f,
                     "noise polluted the cached policy");
    }
}

void test_noise_is_deterministic_and_epsilon_zero_is_a_no_op() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard board = startup_board();
    const std::vector<int> legal = board.getLegalMoveIndices();
    MCTSTestAccess::store(mcts, board, legal, politique_non_uniforme(legal),
                          0.1f);

    MCTSNode temoin(0.0f);
    MCTSNode first(0.0f);
    MCTSNode second(0.0f);
    MCTSNode sans_bruit(0.0f);
    mcts.expand_node_single(&temoin, board);
    mcts.expand_node_single(&first, board);
    mcts.expand_node_single(&second, board);
    mcts.expand_node_single(&sans_bruit, board);
    const std::vector<float> base = priors_des_enfants(temoin);

    MCTSTestAccess::seed_noise(mcts, 42);
    mcts.add_dirichlet_noise(&first, 0.30f);
    MCTSTestAccess::seed_noise(mcts, 42);
    mcts.add_dirichlet_noise(&second, 0.30f);
    mcts.add_dirichlet_noise(&sans_bruit, 0.0f);

    const std::vector<float> p1 = priors_des_enfants(first);
    const std::vector<float> p2 = priors_des_enfants(second);
    const std::vector<float> p3 = priors_des_enfants(sans_bruit);
    require_test(p1 == p2, "the same seed did not reproduce the noise");
    require_test(p3 == base, "epsilon zero changed the priors");
    float somme = 0.0f;
    bool fini = true;
    for (std::size_t i = 0; i < p1.size(); ++i) {
        somme += p1[i];
        if (p1[i] < 0.0f || !std::isfinite(p1[i])) fini = false;
    }
    require_test(std::fabs(somme - 1.0f) < 1e-5f,
                 "noised priors do not sum to one");
    require_test(fini, "noised priors are negative or non-finite");
}

void test_expansion_keeps_moves_beyond_the_cache_capacity() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 16, 0);
    Chessboard board = startup_board();
    MCTSNode node(0.0f);
    PathReservation reservation;
    require_test(reservation.try_claim(&node), "claim failed");

    std::vector<int> legal;
    for (int i = 0; i < TT_MAX_MOVES + 1; ++i) {
        legal.push_back(3 + 5 * i);
    }
    std::vector<float> policy(POLICY_SIZE, 0.0f);
    for (int idx : legal) policy[static_cast<std::size_t>(idx)] = 1.0f;
    const int dominant = legal.back();
    policy[static_cast<std::size_t>(dominant)] = 10.0f;

    const EvaluationCacheKey key = MCTSTestAccess::key_for(mcts, board);
    MCTSTestAccess::expand_prepared(mcts, &node, legal, key, policy.data(),
                                    0.1f, reservation);

    require_test(node.state.load() == NodeState::Expanded,
                 "the prepared expansion did not publish the node");
    require_test(node.children.size() == legal.size(),
                 "the expansion truncated the legal moves");
    float somme = 0.0f;
    bool dominant_trouve = false;
    for (const auto& child : node.children) {
        somme += child.second->prior;
        if (child.first == dominant) {
            dominant_trouve = true;
            require_test(
                std::fabs(child.second->prior - 10.0f / 138.0f) < 1e-5f,
                "the 129th move was normalised on a truncated list");
        }
    }
    require_test(dominant_trouve,
                 "the last legal move is missing from the tree");
    require_test(std::fabs(somme - 1.0f) < 1e-5f,
                 "priors do not sum to one");
    require_test(mcts.inspect_tree(&node).violations == 0,
                 "the 129-child node violates the tree invariants");
}

void test_hot_root_noises_like_a_cold_root() {
    DiscriminatingEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard board = startup_board();

    MCTSNode froid(0.0f);
    mcts.expand_node_single(&froid, board);
    require_test(evaluator.batch_sizes.size() == 1,
                 "la racine froide n'a pas appele le reseau");
    MCTSTestAccess::seed_noise(mcts, 777);
    mcts.add_dirichlet_noise(&froid, 0.30f);

    // Meme position servie par la table : meme politique, meme bruit.
    MCTSNode chaud(0.0f);
    mcts.expand_node_single(&chaud, board);
    require_test(evaluator.batch_sizes.size() == 1,
                 "la racine chaude a rappele le reseau");
    MCTSTestAccess::seed_noise(mcts, 777);
    mcts.add_dirichlet_noise(&chaud, 0.30f);

    const std::vector<float> p_froid = priors_des_enfants(froid);
    const std::vector<float> p_chaud = priors_des_enfants(chaud);
    require_test(p_froid.size() == p_chaud.size(),
                 "les deux racines n'ont pas les memes enfants");
    for (std::size_t i = 0; i < p_froid.size(); ++i) {
        require_test(
            std::fabs(p_froid[i] - p_chaud[i]) < 1e-6f,
            "une racine chaude ne se bruite pas comme une racine froide");
    }
}

void test_a_real_position_beyond_the_cache_capacity_keeps_all_moves() {
    // Position construite par Nenad Petrovic. Fixture verifiee avec
    // python-chess : position legale, 218 coups legaux pour les Blancs, et le
    // moteur en trouve exactement 218.
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.loadFEN(
        "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1");

    const std::vector<int> legal = board.getLegalMoveIndices();
    require_test(legal.size() > static_cast<std::size_t>(TT_MAX_MOVES),
                 "the fixture no longer exceeds the cache capacity");

    MCTSNode root(0.0f);
    mcts.expand_node_single(&root, board);

    require_test(root.children.size() == legal.size(),
                 "the expansion lost legal moves of a real position");
    require_test(evaluator.batch_sizes.size() == 1,
                 "the first expansion did not use the network");
    float somme = 0.0f;
    for (const auto& child : root.children) somme += child.second->prior;
    require_test(std::fabs(somme - 1.0f) < 1e-5f,
                 "priors of the real position do not sum to one");
    require_test(mcts.inspect_tree(&root).violations == 0,
                 "the 218-child node violates the tree invariants");

    // La politique depasse la capacite du cache : la deuxieme racine doit
    // rappeler le reseau au lieu de relire une entree tronquee.
    MCTSNode second(0.0f);
    mcts.expand_node_single(&second, board);
    require_test(evaluator.batch_sizes.size() == 2,
                 "an oversized policy was served from the cache");
    require_test(second.children.size() == legal.size(),
                 "the second expansion lost legal moves");
}

void test_virtual_loss_changes_the_mono_batched_search() {
    // Le chemin mono batché collecte jusqu'à huit feuilles d'affilée dans un
    // seul thread : la feuille suivante voit le virtual loss des précédentes,
    // donc l'amplitude change la recherche. Le self-play, qui descend chaque
    // arbre une seule fois par vague, y est au contraire insensible.
    std::vector<std::vector<MoveStats>> resultats;
    for (int vloss : {1, 2}) {
        DiscriminatingEvaluator evaluator;
        MCTS mcts(&evaluator, 8192, 0);
        mcts.set_fixed_batch(true);
        mcts.set_tuning(SearchTuning{vloss, 0.30f, 4});
        Chessboard board;
        board.loadFEN(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        mcts.step_analysis(board, 700, 1.4f, 8, 1);
        resultats.push_back(mcts.get_analysis_results());
    }

    std::vector<std::pair<int, int>> visites[2];
    for (int k = 0; k < 2; ++k) {
        for (const MoveStats& stats : resultats[static_cast<std::size_t>(k)]) {
            visites[k].emplace_back(stats.move_idx, stats.visits);
        }
        std::sort(visites[k].begin(), visites[k].end());
    }
    require_test(!visites[0].empty(), "no visited move in the mono search");
    require_test(visites[0] != visites[1],
                 "the amplitude did not change the mono batched search");
}

void test_gpu_phase_is_quiet_and_update_root_waits_for_session() {
    ControlledEvaluator evaluator;
    EvaluationGate gate;
    evaluator.before_evaluate = [&]() {
        if (evaluator.batch_sizes.size() == 2) gate.block();
    };
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    const int legal_move = board.getLegalMoveIndices().front();
    std::exception_ptr search_error;

    std::thread search([&]() {
        try {
            mcts.step_analysis(board, 8, 1.4f, 8, 4);
        }
        catch (...) {
            search_error = std::current_exception();
        }
    });
    gate.wait_until_blocked();

    const SearchCounters blocked = mcts.get_counters();
    for (int i = 0; i < 1000; ++i) std::this_thread::yield();
    const SearchCounters still_blocked = mcts.get_counters();
    require_test(blocked.waves == still_blocked.waves
                     && blocked.leaf_collisions
                         == still_blocked.leaf_collisions
                     && blocked.completed_simulations
                         == still_blocked.completed_simulations,
                 "CPU collection continued during evaluator call");

    auto update = std::async(std::launch::async, [&]() {
        mcts.update_root(legal_move);
        return true;
    });
    require_test(update.wait_for(0ms) == std::future_status::timeout,
                 "update_root entered while search held raw node pointers");

    gate.release();
    search.join();
    if (search_error) std::rethrow_exception(search_error);
    require_test(update.wait_for(2s) == std::future_status::ready,
                 "update_root did not resume after search");
    require_test(update.get(), "update_root future failed");
    require_quiescent(mcts.inspect_tree());
}

void test_weighted_virtual_loss_leaves_no_residue() {
    constexpr std::array<int, 4> AMPLITUDES{1, 2, 3, 8};
    constexpr std::array<int, 3> WORKERS{1, 4, 8};
    constexpr std::array<int, 2> BUDGETS{3, 17};

    for (int workers : WORKERS) {
        for (int virtual_loss : AMPLITUDES) {
            ControlledEvaluator evaluator;
            MCTS mcts(&evaluator, 8192, 0);
            mcts.set_tuning(SearchTuning{virtual_loss, 0.30f, 4});
            Chessboard board = startup_board();

            for (int budget : BUDGETS) {
                const TreeReport before = mcts.inspect_tree();
                const SearchCounters counters_before = mcts.get_counters();
                mcts.step_analysis(board, budget, 1.4f, 8, workers);
                const TreeReport after = mcts.inspect_tree();
                const SearchCounters counters_after = mcts.get_counters();

                require_test(
                    after.root_visits - before.root_visits
                        == static_cast<std::uint64_t>(budget),
                    "weighted virtual loss changed the visit budget");
                require_test(
                    counters_after.completed_simulations
                        - counters_before.completed_simulations
                        == static_cast<std::uint64_t>(budget),
                    "completed_simulations does not match budget");
                require_quiescent(after);
                require_test(board.toFEN() == startup_board().toFEN(),
                             "weighted search changed the input board");
            }
        }
    }
}

void test_weighted_reservations_survive_failure_then_recover() {
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 2;
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_tuning(SearchTuning{3, 0.30f, 4});
    Chessboard board = startup_board();

    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 8);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "controlled evaluator failure was swallowed");
    require_quiescent(mcts.inspect_tree());

    evaluator.fail_on_call = 0;
    const TreeReport before = mcts.inspect_tree();
    const SearchCounters counters_before = mcts.get_counters();
    mcts.step_analysis(board, 3, 1.4f, 3, 2);
    const TreeReport after = mcts.inspect_tree();
    const SearchCounters counters_after = mcts.get_counters();

    require_quiescent(after);
    require_test(after.root_visits - before.root_visits == 3,
                 "search did not recover after weighted failure");
    require_test(
        counters_after.completed_simulations
            - counters_before.completed_simulations == 3,
        "recovery did not complete the budget");
}

void test_invalid_tail_of_batch_is_rejected_before_any_publication() {
    ControlledEvaluator evaluator;
    evaluator.corrupt_call = 2;
    evaluator.values_corruption =
        ControlledEvaluator::Corruption::NonFinite;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();

    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 4);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "invalid batch tail was accepted");
    const TreeReport report = mcts.inspect_tree();
    require_quiescent(report);
    require_test(report.root_visits == 0,
                 "invalid wave performed a backup before rejection");

    evaluator.corrupt_call = 0;
    mcts.step_analysis(board, 3, 1.4f, 3, 2);
    const TreeReport recovered = mcts.inspect_tree();
    require_quiescent(recovered);
    require_test(recovered.root_visits == 3,
                 "search did not recover after a rejected batch tail");
}

}  // namespace

int main() {
    try {
        test_budgets_and_invariants_across_configurations();
        test_terminal_root_uses_root_visits();
        test_evaluator_failure_cleans_session_and_allows_recovery();
        test_invalid_evaluator_outputs_clean_session();
        test_fixed_batch_pads_wave_calls_and_keeps_budget();
        test_fixed_batch_pads_mono_batch();
        test_tuning_defaults_validation_and_propagation();
        test_weighted_virtual_loss_leaves_no_residue();
        test_weighted_reservations_survive_failure_then_recover();
        test_invalid_tail_of_batch_is_rejected_before_any_publication();
        test_each_leaf_gets_the_value_and_priors_of_its_own_tensor();
        test_terminal_leaves_do_not_shift_network_rows();
        test_partial_waves_pad_to_the_fixed_shape();
        test_terminal_root_never_sends_a_wave();
        test_padding_sentinels_are_ignored_and_counters_stay_honest();
        test_the_association_check_detects_a_permutation();
        test_mate_at_hundred_is_a_loss_in_every_search_path();
        test_scalar_search_restores_the_board_after_an_evaluator_error();
        test_mate_delivered_at_hundred_from_ninety_nine();
        test_terminal_classification_prefers_mate_over_rule_draws();
        test_repetition_is_a_draw_in_the_search();
        test_tt_hit_expands_the_root_and_does_not_pollute_the_cache();
        test_noise_is_deterministic_and_epsilon_zero_is_a_no_op();
        test_hot_root_noises_like_a_cold_root();
        test_expansion_keeps_moves_beyond_the_cache_capacity();
        test_a_real_position_beyond_the_cache_capacity_keeps_all_moves();
        test_virtual_loss_changes_the_mono_batched_search();
        test_gpu_phase_is_quiet_and_update_root_waits_for_session();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
