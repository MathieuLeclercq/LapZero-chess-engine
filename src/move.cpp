#include "move.hpp"



//...............Constructors...............
Move::Move(Square orig_square, Square dest_square, PieceType promotion)
{
    this->piece = orig_square.getPiece();
    this->orig_square = orig_square;
    this->dest_square = dest_square;
    this->promotion = promotion;
}

//...............Getters...............
const Piece& Move::getPiece() const
{
    return this->piece;
}

Piece& Move::getPiece()
{
    return this->piece;
}

const Square& Move::getOrigSquare() const
{
    return this->orig_square;
}

Square& Move::getOrigSquare()
{
    return this->orig_square;
}

const Square& Move::getDestSquare() const
{
    return this->dest_square;
}

Square& Move::getDestSquare()
{
    return this->dest_square;
}

PieceType Move::getPromotion() const
{
    return this->promotion;
}

bool Move::operator==(const Move& other) const
{
    return (this->orig_square == other.orig_square &&
        this->dest_square == other.dest_square &&
        this->promotion == other.promotion);
}