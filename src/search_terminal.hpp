#pragma once

#include "chessboard.hpp"

// Valeur terminale du point de vue du camp au trait : mat = -1, pat et nulles
// de regle = 0. Le mat prime quand les conditions se recouvrent, par exemple un
// mat porte au centieme demi-coup calme : sans ce controle, la position serait
// notee nulle alors que le roi est mate.
//
// A n'appeler que sur une position terminale. Une position en echec avec des
// coups legaux rendrait 0, qui est la valeur d'une nulle et non d'une position
// vivante. Le chemin rapide evite toute generation de coups quand le roi n'est
// pas en echec.
inline float terminal_value_for(Chessboard& board) {
    if (!board.isInCheck()) {
        return 0.0f;
    }
    return board.hasAnyLegalMove() ? 0.0f : -1.0f;
}
