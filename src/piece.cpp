#include "piece.hpp"


//...............Constructors...............

Piece::Piece()
{
    m_type = NONE;
    m_color = NO_COLOR;
    m_value = 0;
}

Piece::Piece(Color color, PieceType type)
{
    m_color = color;
    m_type = type;
    this->setValue();
}


//...............Getters...............
const PieceType& Piece::getType() const
{
    return m_type;
}

Color Piece::getColor()
{
    return m_color;
}

const Color& Piece::getColor() const
{
    return m_color;
}

int Piece::getValue() const
{
    int value = m_value;
    return value;
}

bool Piece::operator==(const Piece& other) const
{
    return (m_type == other.m_type && m_color == other.m_color);
}

//...............Setters...............
void Piece::setType(PieceType type)
{
    m_type = type;

}



void Piece::setValue()
{
    if (m_type == KING)
        m_value = 2; //  king : no sense of value
    else if (m_type == QUEEN)
        m_value = 9;
    else if (m_type == ROOK)
        m_value = 5;
    else if (m_type == BISHOP || m_type == KNIGHT)
        m_value = 3;
    else if (m_type == PAWN)
        m_value = 1;
    else
        m_value = 0;
}

