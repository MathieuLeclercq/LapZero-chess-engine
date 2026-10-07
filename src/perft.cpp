#include "perft.hpp"

#include <string>
#include <unordered_set>

namespace {

void report_violation(PerftReport& report, const std::string& message) {
    report.violations++;
    if (report.messages.size() < MAX_PERFT_MESSAGES) {
        report.messages.push_back(message);
    }
}

void run_strict_checks(Chessboard& board,
                       const std::vector<Move>& moves,
                       const PerftOptions& opts,
                       PerftReport& report) {
    // Trou 3 : hasAnyLegalMove duplique la boucle interne de
    // getLegalMovesForSquare. On confronte la copie à son original.
    const bool has_any = board.hasAnyLegalMove();
    if (has_any != !moves.empty()) {
        report_violation(report,
            "hasAnyLegalMove=" + std::string(has_any ? "true" : "false")
            + " mais getAllLegalMoves en renvoie " + std::to_string(moves.size()));
    }

    // Trou 1 : la couche d'encodage.
    const std::vector<int> indices = board.getLegalMoveIndices();

    if (indices.size() != moves.size()) {
        report_violation(report,
            "encodeMove : " + std::to_string(indices.size()) + " indice(s) pour "
            + std::to_string(moves.size()) + " coup(s) legaux");
    }

    std::unordered_set<int> seen;
    seen.reserve(indices.size() * 2);
    for (int index : indices) {
        if (index < 0 || index > 4671) {
            report_violation(report,
                "indice hors bornes [0, 4671] : " + std::to_string(index));
        }
        if (!seen.insert(index).second) {
            report_violation(report,
                "indice duplique, deux coups encodes pareil : " + std::to_string(index));
        }
        // Trou 2 : le decodage. Le seul controle qui exerce decodeMoveIndex,
        // que le parcours utilise sinon sans le confronter a son inverse.
        if (opts.roundtrip) {
            const int reencode = board.encodeMove(board.decodeMoveIndex(index));
            if (reencode != index) {
                report_violation(report,
                    "roundtrip : l'index " + std::to_string(index)
                    + " se reencode en " + std::to_string(reencode));
            }
        }
    }
}

void run_fen_check(Chessboard& board, PerftReport& report) {
    const std::string fen = board.toFEN();

    Chessboard probe;
    probe.loadFEN(fen);

    if (probe.getZobristHash() != board.getZobristHash()) {
        report_violation(report, "toFEN/loadFEN incoherent, hash different : " + fen);
    }
}

uint64_t perft_rec(Chessboard& board, int depth,
                   const PerftOptions& opts, PerftReport& report) {
    if (depth == 0) return 1;

    std::vector<Move> moves = board.getAllLegalMoves();

    if (opts.strict || opts.roundtrip) run_strict_checks(board, moves, opts, report);
    if (opts.check_fen) run_fen_check(board, report);

    uint64_t nodes = 0;
    for (const Move& move : moves) {
        const int orig_f = move.getOrigSquare().getFile();
        const int orig_r = move.getOrigSquare().getRank();
        const int dest_f = move.getDestSquare().getFile();
        const int dest_r = move.getDestSquare().getRank();

        // Un refus ici signifie que movePiece rejette un coup que le moteur
        // vient lui-même de générer : c'est un bug, pas un cas normal.
        if (!board.movePiece(orig_f, orig_r, dest_f, dest_r, move.getPromotion(), false)) {
            report_violation(report,
                "movePiece a refuse un coup genere : " + move_to_uci(move));
            continue;
        }

        nodes += perft_rec(board, depth - 1, opts, report);
        board.undoMove();
    }
    return nodes;
}

} // namespace

std::string move_to_uci(const Move& move) {
    std::string uci;
    uci += static_cast<char>('a' + move.getOrigSquare().getFile());
    uci += static_cast<char>('1' + move.getOrigSquare().getRank());
    uci += static_cast<char>('a' + move.getDestSquare().getFile());
    uci += static_cast<char>('1' + move.getDestSquare().getRank());

    switch (move.getPromotion()) {
    case QUEEN:  uci += 'q'; break;
    case ROOK:   uci += 'r'; break;
    case BISHOP: uci += 'b'; break;
    case KNIGHT: uci += 'n'; break;
    default: break;
    }
    return uci;
}

PerftReport perft(Chessboard& board, int depth, const PerftOptions& opts) {
    PerftReport report;
    report.nodes = perft_rec(board, depth, opts, report);
    return report;
}

std::vector<std::pair<std::string, uint64_t>> perft_divide(
    Chessboard& board, int depth, const PerftOptions& opts,
    PerftReport* report) {

    std::vector<std::pair<std::string, uint64_t>> result;
    if (depth <= 0) return result;

    PerftReport fallback;
    PerftReport& sink = report ? *report : fallback;

    std::vector<Move> moves = board.getAllLegalMoves();
    result.reserve(moves.size());

    // La position racine doit etre controlee elle aussi : perft_rec ne voit
    // que ses descendants.
    if (opts.strict || opts.roundtrip) run_strict_checks(board, moves, opts, sink);
    if (opts.check_fen) run_fen_check(board, sink);

    for (const Move& move : moves) {
        const int orig_f = move.getOrigSquare().getFile();
        const int orig_r = move.getOrigSquare().getRank();
        const int dest_f = move.getDestSquare().getFile();
        const int dest_r = move.getDestSquare().getRank();

        if (!board.movePiece(orig_f, orig_r, dest_f, dest_r, move.getPromotion(), false)) {
            report_violation(sink,
                "movePiece a refuse un coup genere : " + move_to_uci(move));
            result.emplace_back(move_to_uci(move), 0);
            continue;
        }

        result.emplace_back(move_to_uci(move), perft_rec(board, depth - 1, opts, sink));
        board.undoMove();
    }
    return result;
}
