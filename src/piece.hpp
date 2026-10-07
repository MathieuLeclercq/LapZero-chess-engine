#pragma once
#include <iostream>

enum PieceType 
{
    PAWN,   // 0
    KNIGHT, // 1
    BISHOP, // 2
    ROOK,   // 3
    QUEEN,  // 4
    KING,   // 5
    NONE    // 6
};

enum Color
{
    WHITE,
    BLACK,
    NO_COLOR
};

class Piece
{
    private:
        PieceType m_type;
        Color m_color;
        int m_value;

    public:
        // constructors
        Piece();
        Piece(Color color, PieceType type);


        // getters
        const PieceType& getType() const;
        const Color& getColor() const;
        Color getColor();
        int getValue() const;
        inline int Piece::getZobristIndex() const {
            if (m_type == NONE) return -1;
            return static_cast<int>(m_type) + (m_color == BLACK ? 6 : 0);
        }
        bool operator==(const Piece& other) const;

        // setters
        void setType(PieceType type);
        void setValue();

};
