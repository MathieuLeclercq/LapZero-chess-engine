#include "chessboard.hpp"
#include "zobrist.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint64_t CURRENT_CONTEXT_DOMAIN = 0xA25B7C3D4E6F8101ULL;
constexpr uint64_t HISTORY_DOMAIN = 0xB36C8D4E5F710212ULL;
constexpr uint64_t COMBINED_DOMAIN = 0xC47D9E5F60812323ULL;
constexpr uint64_t HISTORY_SLOT_DOMAIN = 0xD58EAF6071923434ULL;
constexpr uint64_t EMPTY_HISTORY_SLOT = 0xE69FB07182A34545ULL;

uint64_t melanger(uint64_t seed, uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;
    return seed ^ (value + 0x9E3779B97F4A7C15ULL
                   + (seed << 6) + (seed >> 2));
}

uint8_t categorie_repetition(const std::vector<uint64_t>& hashes,
                             size_t index) {
    int occurrences = 0;
    for (size_t i = 0; i <= index; ++i) {
        if (hashes[i] == hashes[index]) {
            ++occurrences;
        }
    }
    if (occurrences >= 3) return 2;
    if (occurrences == 2) return 1;
    return 0;
}

int indice_roque(const StateSnapshot& snapshot) {
    return (snapshot.short_castle_white ? 1 : 0)
        | (snapshot.long_castle_white ? 2 : 0)
        | (snapshot.short_castle_black ? 4 : 0)
        | (snapshot.long_castle_black ? 8 : 0);
}

}  // namespace

EvaluationCacheKey Chessboard::getEvaluationCacheKey(int history_depth) const {
    if (history_depth < 0 || history_depth > 7) {
        throw std::invalid_argument(
            "getEvaluationCacheKey : history_depth doit etre compris entre 0 et 7");
    }

    std::vector<uint64_t> all_hashes;
    all_hashes.reserve(m_snapshotHistory.size() + 1);
    for (const StateSnapshot& snapshot : m_snapshotHistory) {
        all_hashes.push_back(snapshot.zobrist_hash);
    }
    all_hashes.push_back(m_current_zobrist_hash);

    EvaluationCacheKey key;
    key.position_hash = m_current_zobrist_hash;
    key.half_move_clock = static_cast<uint16_t>(m_half_move_clock);
    key.repetition_category = categorie_repetition(
        all_hashes, all_hashes.size() - 1);
    key.total_moves_bucket = static_cast<uint8_t>(
        std::min<size_t>(100, m_boardHistory.size() / 2));

    key.current_context_hash = CURRENT_CONTEXT_DOMAIN;
    key.current_context_hash = melanger(
        key.current_context_hash, key.position_hash);
    key.current_context_hash = melanger(
        key.current_context_hash, key.half_move_clock);
    key.current_context_hash = melanger(
        key.current_context_hash, key.repetition_category);
    key.current_context_hash = melanger(
        key.current_context_hash, key.total_moves_bucket);

    key.history_hash = HISTORY_DOMAIN;
    for (int t = 1; t <= history_depth; ++t) {
        key.history_hash = melanger(
            key.history_hash, HISTORY_SLOT_DOMAIN ^ static_cast<uint64_t>(t));

        const int history_index =
            static_cast<int>(m_boardHistory.size()) - 1 - t;
        if (m_amnesia_mode || history_index < 0) {
            key.history_hash = melanger(
                key.history_hash, EMPTY_HISTORY_SLOT);
            continue;
        }

        const StateSnapshot& snapshot =
            m_snapshotHistory[static_cast<size_t>(history_index)];
        uint64_t placement_hash = snapshot.zobrist_hash;
        placement_hash ^= Zobrist::CASTLING_KEYS[indice_roque(snapshot)];
        if (snapshot.en_passant && snapshot.en_passant_file >= 0
            && snapshot.en_passant_file < 8) {
            placement_hash ^=
                Zobrist::EN_PASSANT_KEYS[snapshot.en_passant_file];
        }

        const bool historical_black_to_move =
            (t % 2 == 0) ? (m_turn == BLACK) : (m_turn == WHITE);
        if (historical_black_to_move) {
            placement_hash ^= Zobrist::BLACK_TO_MOVE;
        }

        key.history_hash = melanger(key.history_hash, placement_hash);
        key.history_hash = melanger(
            key.history_hash,
            categorie_repetition(all_hashes,
                                  static_cast<size_t>(history_index)));
    }

    key.combined_hash = COMBINED_DOMAIN;
    key.combined_hash = melanger(
        key.combined_hash, key.current_context_hash);
    key.combined_hash = melanger(key.combined_hash, key.history_hash);
    key.combined_hash = melanger(
        key.combined_hash, static_cast<uint64_t>(history_depth));
    return key;
}
