#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "chessboard.hpp"

// Options de contrôle activables pendant le parcours.
// Elles n'affectent pas le comptage de noeuds, seulement les vérifications.
struct PerftOptions {
    bool strict = false;    // trous 1 et 3 : encodage et hasAnyLegalMove
    bool check_fen = false; // aller-retour toFEN / loadFEN (coûteux, faible profondeur)
    bool roundtrip = false; // trou 2 : encodeMove(decodeMoveIndex(i)) == i
};

struct PerftReport {
    uint64_t nodes = 0;
    uint64_t violations = 0;
    std::vector<std::string> messages; // tronqué à MAX_PERFT_MESSAGES
};

// Nombre maximum de messages conservés, pour éviter de noyer la sortie
// quand un bug se déclenche sur des millions de positions.
constexpr size_t MAX_PERFT_MESSAGES = 20;

// Compte les feuilles de l'arbre des coups légaux à la profondeur donnée.
PerftReport perft(Chessboard& board, int depth, const PerftOptions& opts = {});

// Compte les feuilles par coup racine. Le premier membre de chaque paire
// est le coup en notation UCI (ex: "e2e4", "a7a8q").
// Si report est fourni, les violations detectees pendant le parcours y sont
// accumulees. Sans lui, elles seraient perdues.
std::vector<std::pair<std::string, uint64_t>> perft_divide(
    Chessboard& board, int depth, const PerftOptions& opts = {},
    PerftReport* report = nullptr);

std::string move_to_uci(const Move& move);
