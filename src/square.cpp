#include "square.hpp"

//...............Constructors...............

Square::Square()
{
    // set the file and rank to -1
    this->file = -1;
    this->rank = -1;
    // this->isOccupied = false;

}

Square::Square(int file, int rank)
{
    this->file = file;
    this->rank = rank;
    // this->isOccupied = false;
}

Square::Square(int file, int rank, Piece piece)
{
    this->file = file;
    this->rank = rank;
    this->piece = piece;
    // this->isOccupied = true;
}

//...............Getters...............

const Piece& Square::getPiece() const
{
    return this->piece;
}

Piece& Square::getPiece()
{
    return this->piece;
}

bool Square::checkOccupied() const
{
    return this->piece.getType() != NONE;
}

int Square::getFile() const
/*
Returns an integer between 0 and 7 representing the file of the square
*/
{
    return this->file;
}


int Square::getRank() const
/*
Returns an integer between + and 7 representing the rank of the square
*/
{
    return this->rank;
}


std::string Square::getName() const
{
    // Sécurité au cas où la case n'est pas initialisée (file/rank à -1)
    if (this->file < 0 || this->rank < 0)
    {
        return "Invalid";
    }

    std::string square_name = "";
    // On convertit l'index (0-7) en caractère ASCII ('a'-'h' et '1'-'8')
    square_name += (char)('a' + this->file);
    square_name += (char)('1' + this->rank);

    return square_name;
}



//...............Overload Operators...............

Square& Square::operator=(const Square& other)
{

    if (this != &other)
    {
        this->file = other.file;
        this->rank = other.rank;
        this->piece = other.piece;
    }

    return *this;
}

bool Square::operator == (const Square& square) const
{
    return (this->file == square.file &&
        this->rank == square.rank &&
        this->piece == square.piece);
}

//...............Setters...............

void Square::setPiece(Piece piece)
{
    this->piece = piece;
    // this->isOccupied = true;
}

void Square::setPosition(int file, int rank)
{
    this->file = file;
    this->rank = rank;
}

void Square::removePiece()
{
    this->piece = Piece();
}

