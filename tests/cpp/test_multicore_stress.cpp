#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<const char*, 3> SEARCH_FENS{
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
};

Chessboard board_from_fen(const char* fen) {
    Chessboard board;
    board.loadFEN(fen);
    return board;
}

void require_quiescent(const TreeReport& report) {
    require_test(report.en_vol == 0, "tree retained in-flight nodes");
    require_test(report.pending == 0, "tree retained Pending nodes");
    require_test(report.violations == 0, "tree invariant failed");
}

void require_policy(const std::vector<float>& policy) {
    require_test(policy.size() == 4672, "policy has the wrong size");
    require_test(std::all_of(policy.begin(), policy.end(), [](float value) {
        return std::isfinite(value) && value >= 0.0f;
    }), "policy contains an invalid probability");
    const float sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
    require_test(std::fabs(sum - 1.0f) < 1e-4f,
                 "policy probabilities do not sum to one");
}

void test_repeated_fresh_and_reused_searches() {
    constexpr std::array<int, 3> WORKERS{2, 4, 8};
    constexpr std::array<int, 2> BATCHES{1, 8};

    for (const char* fen : SEARCH_FENS) {
        for (int workers : WORKERS) {
            for (int batch : BATCHES) {
                ControlledEvaluator evaluator;
                MCTS mcts(&evaluator, 8192, 0);
                Chessboard board = board_from_fen(fen);
                const std::string original = board.toFEN();

                for (int repetition = 0; repetition < 100; ++repetition) {
                    const SearchCounters before = mcts.get_counters();
                    if ((repetition % 2) == 0) {
                        require_policy(mcts.mcts_search(
                            board, 64, 1.4f, false, batch, workers));
                    }
                    else {
                        const std::uint64_t visits_before =
                            mcts.inspect_tree().root_visits;
                        mcts.step_analysis(
                            board, 64, 1.4f, batch, workers);
                        const TreeReport report = mcts.inspect_tree();
                        require_test(
                            report.root_visits - visits_before == 64,
                            "reused root completed the wrong budget");
                        require_quiescent(report);
                    }
                    const SearchCounters after = mcts.get_counters();
                    require_test(
                        after.completed_simulations
                            - before.completed_simulations == 64,
                        "fresh or reused search lost simulations");
                    require_test(board.toFEN() == original,
                                 "stress search changed the input board");
                }
            }
        }
    }
}

void test_worker_count_can_change_on_a_live_pool() {
    constexpr std::array<int, 4> WORKERS{8, 2, 1, 4};
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = board_from_fen(SEARCH_FENS[0]);

    std::uint64_t expected_visits = 0;
    for (int workers : WORKERS) {
        mcts.step_analysis(board, 64, 1.4f, 8, workers);
        expected_visits += 64;
        const TreeReport report = mcts.inspect_tree();
        require_test(report.root_visits == expected_visits,
                     "worker-count transition lost visits");
        require_quiescent(report);
    }
}

Chessboard repetition_board() {
    Chessboard board;
    board.setStartupPieces();
    for (const char* move : {
             "g1f3", "g8f6", "f3g1", "f6g8",
             "g1f3", "g8f6", "f3g1", "f6g8"}) {
        require_test(board.movePieceUCI(move),
                     "could not build repetition fixture");
    }
    return board;
}

void require_terminal_search(Chessboard board, float expected_q) {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    mcts.step_analysis(board, 17, 1.4f, 8, 4);
    const TreeReport report = mcts.inspect_tree();
    require_test(report.root_visits == 17,
                 "terminal root completed the wrong budget");
    require_test(report.nodes == 1, "terminal root grew children");
    require_test(std::fabs(mcts.get_root_q() - expected_q) < 1e-6f,
                 "terminal root has the wrong value");
    require_quiescent(report);
}

void test_terminal_rules_under_multicore_search() {
    require_terminal_search(board_from_fen(
        "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
        -1.0f);
    require_terminal_search(board_from_fen(
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), 0.0f);
    require_terminal_search(board_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1"),
        0.0f);
    require_terminal_search(board_from_fen(
        "8/8/8/8/8/8/7k/K7 w - - 0 1"), 0.0f);
    require_terminal_search(repetition_board(), 0.0f);
}

void test_value_sign_alternates_at_successive_depths() {
    ControlledEvaluator evaluator;
    evaluator.output_value = 0.25f;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = board_from_fen(
        "8/8/8/8/8/k7/8/KR6 b - - 0 1");
    require_test(board.getLegalMoveIndices().size() == 1,
                 "sign fixture must have one legal move");

    mcts.step_analysis(board, 1, 1.4f, 1, 2);
    require_test(std::fabs(mcts.get_root_q() + 0.25f) < 1e-6f,
                 "one-ply value has the wrong sign");

    mcts.step_analysis(board, 1, 1.4f, 1, 2);
    require_test(std::fabs(mcts.get_root_q()) < 1e-6f,
                 "two-ply value did not alternate sign");
    require_quiescent(mcts.inspect_tree());
}

}  // namespace

int main() {
    try {
        test_repeated_fresh_and_reused_searches();
        test_worker_count_can_change_on_a_live_pool();
        test_terminal_rules_under_multicore_search();
        test_value_sign_alternates_at_successive_depths();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
