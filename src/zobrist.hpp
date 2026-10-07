#pragma once
#include <cstdint>

namespace Zobrist {
    // 64 cases * 12 types de pièces (6 blanc, 6 noir)
    extern const uint64_t PIECE_KEYS[64][12];

    // Trait aux noirs
    extern const uint64_t BLACK_TO_MOVE;

    // Droits de roque (16 possibilités : de 0000 à 1111)
    extern const uint64_t CASTLING_KEYS[16];

    // Case en passant (8 colonnes possibles)
    extern const uint64_t EN_PASSANT_KEYS[8];

}