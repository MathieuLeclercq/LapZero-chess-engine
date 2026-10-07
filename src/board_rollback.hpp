#pragma once

#include "chessboard.hpp"

// Restaure le plateau a la sortie du bloc, exception comprise. Chaque descente
// joue des coups avant de les defaire : sans ce garde, une erreur au milieu de
// la descente laisserait le plateau sur la feuille alors que l'arbre pointe
// toujours la racine.
class BoardRollback {
public:
    explicit BoardRollback(Chessboard& board) : m_board(board) {}

    BoardRollback(const BoardRollback&) = delete;
    BoardRollback& operator=(const BoardRollback&) = delete;

    ~BoardRollback() {
        while (m_moves_played > 0) {
            m_board.undoMove();
            --m_moves_played;
        }
    }

    void move_played() noexcept { ++m_moves_played; }

    // Confie la restauration a l'appelant, qui garde le plateau sur la feuille
    // le temps de lire le tenseur et les coups legaux.
    int release() noexcept {
        const int moves_played = m_moves_played;
        m_moves_played = 0;
        return moves_played;
    }

private:
    Chessboard& m_board;
    int m_moves_played = 0;
};
