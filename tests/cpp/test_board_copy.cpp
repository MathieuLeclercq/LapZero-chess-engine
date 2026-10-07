#include "chessboard.hpp"
#include "test_support.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_same_key(const EvaluationCacheKey& left,
                      const EvaluationCacheKey& right) {
    require_test(left.position_hash == right.position_hash,
                 "position hash differs after copy");
    require_test(left.current_context_hash == right.current_context_hash,
                 "current context differs after copy");
    require_test(left.history_hash == right.history_hash,
                 "history hash differs after copy");
    require_test(left.combined_hash == right.combined_hash,
                 "combined hash differs after copy");
    require_test(left.half_move_clock == right.half_move_clock,
                 "half-move clock differs after copy");
    require_test(left.repetition_category == right.repetition_category,
                 "repetition category differs after copy");
    require_test(left.total_moves_bucket == right.total_moves_bucket,
                 "move-count bucket differs after copy");
}

void require_same_board(const Chessboard& left, const Chessboard& right) {
    require_test(left.toFEN() == right.toFEN(), "FEN differs after copy");
    require_test(left.getZobristHash() == right.getZobristHash(),
                 "Zobrist differs after copy");
    require_test(left.getGameState() == right.getGameState(),
                 "game state differs after copy");
    require_test(left.getMoveHistory() == right.getMoveHistory(),
                 "move history differs after copy");
    require_test(left.getBoardHistory() == right.getBoardHistory(),
                 "board history differs after copy");
    require_same_key(left.getEvaluationCacheKey(7),
                     right.getEvaluationCacheKey(7));

    std::vector<float> left_tensor;
    std::vector<float> right_tensor;
    left.getAlphaZeroTensor(left_tensor);
    right.getAlphaZeroTensor(right_tensor);
    require_test(left_tensor == right_tensor, "tensor differs after copy");
}

void play_line(Chessboard& board, const std::vector<std::string>& moves) {
    for (const std::string& move : moves) {
        require_test(board.movePieceUCI(move), "fixture contains illegal move");
    }
}

void mutate_then_undo(Chessboard& copy) {
    const std::vector<Move> legal = copy.getAllLegalMoves();
    require_test(!legal.empty(), "fixture ended in a terminal position");
    const Move& move = legal.front();
    require_test(copy.movePiece(
        move.getOrigSquare().getFile(), move.getOrigSquare().getRank(),
        move.getDestSquare().getFile(), move.getDestSquare().getRank(),
        move.getPromotion()), "legal fixture move was rejected");
    copy.undoMove();
}

void check_start_position_line(const std::vector<std::string>& moves) {
    Chessboard original;
    original.setStartupPieces();
    play_line(original, moves);

    Chessboard copy = original;
    require_same_board(original, copy);
    mutate_then_undo(copy);
    require_same_board(original, copy);
}

void test_castling_history_copy() {
    check_start_position_line({
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5",
        "a7a6", "b5a4", "g8f6", "e1g1",
    });
}

void test_en_passant_history_copy() {
    check_start_position_line({"e2e4", "a7a6", "e4e5", "d7d5", "e5d6"});
}

void test_promotion_history_copy() {
    Chessboard original;
    original.loadFEN("7k/P7/8/8/8/8/8/7K w - - 0 1");
    play_line(original, {"a7a8q"});

    Chessboard copy = original;
    require_same_board(original, copy);
    mutate_then_undo(copy);
    require_same_board(original, copy);
}

void test_repetition_history_copy() {
    check_start_position_line({"g1f3", "g8f6", "f3g1", "f6g8"});
}

}  // namespace

int main() {
    try {
        test_castling_history_copy();
        test_en_passant_history_copy();
        test_promotion_history_copy();
        test_repetition_history_copy();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
