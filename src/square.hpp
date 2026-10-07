#pragma once
#include <string>
#include "piece.hpp"

class Square
{
private:
    int file;
    int rank;
    Piece piece;

public:
    // constructors
    Square();
    Square(int file, int rank);
    Square(int file, int rank, Piece piece);

    // getters
    const Piece& getPiece() const;
    Piece& getPiece();
    bool checkOccupied() const;
    int getFile() const;
    int getRank() const;
    std::string getName() const;

    // overload operators
    Square& operator=(const Square& other);
    bool operator == (const Square& square) const;

    // setters
    void setPiece(Piece piece);
    void setPosition(int file, int rank);
    void removePiece();
};