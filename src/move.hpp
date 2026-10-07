#pragma once

#include <iostream>
#include "piece.hpp"
#include "square.hpp"

class Move
{
private:
    Piece piece;
    Square orig_square;
    Square dest_square;
    PieceType promotion;


public:
    Move(Square orig_square, Square dest_square, PieceType promotion = NONE);
    const Piece& getPiece() const;
    Piece& getPiece();
    const Square& getOrigSquare() const;
    Square& getOrigSquare();
    const Square& getDestSquare() const;
    Square& getDestSquare();
    PieceType getPromotion() const;

    bool operator==(const Move& other) const;
};