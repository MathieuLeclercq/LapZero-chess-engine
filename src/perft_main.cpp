#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "chessboard.hpp"
#include "perft.hpp"

namespace {

const char* STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct RefPosition {
    const char* name;
    const char* fen;
    std::vector<uint64_t> counts; // counts[i] = perft(i + 1)
    int fast_depth;               // profondeur figee du palier rapide
};

// FEN et comptes verifies le 2026-08-07 contre
// https://www.chessprogramming.org/Perft_Results
// Note : le CPW donne Kiwipete sans les champs demi-coups et coup complet.
// Les "0 1" ajoutes ici designent la meme position.
const std::vector<RefPosition>& reference_positions() {
    static const std::vector<RefPosition> positions = {
        {"1. Depart", STARTPOS_FEN,
         {20, 400, 8902, 197281, 4865609, 119060324}, 5},
        {"2. Kiwipete",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         {48, 2039, 97862, 4085603, 193690690}, 4},
        {"3. Finale",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
         {14, 191, 2812, 43238, 674624, 11030083}, 6},
        {"4. Promotions",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
         {6, 264, 9467, 422333, 15833292}, 5},
        {"5. Talkchess",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
         {44, 1486, 62379, 2103487, 89941194}, 4},
        {"6. Edwards",
         "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
         {46, 2079, 89890, 3894594, 164075551}, 4},
    };
    return positions;
}

void load_position(Chessboard& board, const std::string& fen) {
    board.loadFEN(fen == "startpos" ? STARTPOS_FEN : fen);
}

void print_messages(const PerftReport& report) {
    for (const std::string& message : report.messages) {
        std::cout << "    [VIOLATION] " << message << "\n";
    }
    if (report.violations > report.messages.size()) {
        std::cout << "    [VIOLATION] ... et "
                  << (report.violations - report.messages.size())
                  << " autre(s), non affichee(s).\n";
    }
}

int cmd_divide(const std::string& fen, int depth, const PerftOptions& opts) {
    Chessboard board;
    load_position(board, fen);

    PerftReport report;
    const auto start = std::chrono::steady_clock::now();
    const auto entries = perft_divide(board, depth, opts, &report);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    uint64_t total = 0;
    for (const auto& entry : entries) {
        std::cout << entry.first << ": " << entry.second << "\n";
        total += entry.second;
    }

    std::cout << "\n";
    print_messages(report);
    if (report.violations > 0) {
        std::cout << "Violations   : " << report.violations << "\n";
    }
    std::cout << "Coups racine : " << entries.size() << "\n";
    std::cout << "Total        : " << total << "\n";
    std::cout << "Temps        : " << std::fixed << std::setprecision(2) << seconds << " s\n";
    if (seconds > 0.0) {
        std::cout << "Vitesse      : "
                  << static_cast<uint64_t>(total / seconds) << " noeuds/s\n";
    }
    return report.violations > 0 ? 1 : 0;
}

// Execute chaque position de reference jusqu'a max_depth (bornee par les
// donnees disponibles) et compare aux valeurs attendues.
// En mode calibre, chaque position s'arrete a sa profondeur figee du palier
// rapide, environ 43,5 M de noeuds, au lieu d'une borne commune.
// Retourne 0 si tout concorde, 1 sinon.
int run_campaign(int max_depth, const PerftOptions& opts, const char* label,
                 bool calibrated = false) {
    std::cout << "=== " << label << " ===\n\n";

    bool all_ok = true;
    uint64_t grand_total = 0;
    const auto campaign_start = std::chrono::steady_clock::now();

    for (const RefPosition& position : reference_positions()) {
        std::cout << position.name << "\n";

        const int depth_limit = calibrated
            ? std::min(position.fast_depth,
                       static_cast<int>(position.counts.size()))
            : std::min(max_depth, static_cast<int>(position.counts.size()));

        for (int depth = 1; depth <= depth_limit; ++depth) {
            Chessboard board;
            board.loadFEN(position.fen);

            const auto start = std::chrono::steady_clock::now();
            const PerftReport report = perft(board, depth, opts);
            const double seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();

            const uint64_t expected = position.counts[depth - 1];
            const bool ok = (report.nodes == expected) && (report.violations == 0);
            if (!ok) all_ok = false;
            grand_total += report.nodes;

            std::cout << "  d" << depth
                      << "  attendu " << std::setw(12) << expected
                      << "  obtenu " << std::setw(12) << report.nodes
                      << "  " << (ok ? "OK" : "ECHEC")
                      << "  (" << std::fixed << std::setprecision(2) << seconds << " s";
            if (seconds > 0.01) {
                std::cout << ", " << static_cast<uint64_t>(report.nodes / seconds)
                          << " n/s";
            }
            std::cout << ")\n";

            if (report.nodes != expected) {
                const int64_t delta =
                    static_cast<int64_t>(report.nodes) - static_cast<int64_t>(expected);
                std::cout << "    [ECART] " << (delta > 0 ? "+" : "") << delta
                          << " noeud(s). FEN : " << position.fen << "\n";
            }
            print_messages(report);
        }
        std::cout << "\n";
    }

    const double total_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - campaign_start).count();

    std::cout << "Total noeuds : " << grand_total << "\n";
    std::cout << "Temps total  : " << std::fixed << std::setprecision(2)
              << total_seconds << " s\n";
    if (total_seconds > 0.0) {
        std::cout << "Vitesse      : "
                  << static_cast<uint64_t>(grand_total / total_seconds) << " noeuds/s\n";
    }
    std::cout << "\nResultat : " << (all_ok ? "SUCCES" : "ECHEC") << "\n";
    return all_ok ? 0 : 1;
}

void print_usage() {
    std::cout <<
        "Usage :\n"
        "  chess_perft bench     [--strict]           profondeur 1 a 3, rapide\n"
        "  chess_perft roundtrip                     encodage et decodage, profondeur 1 a 3\n"
        "  chess_perft calibre   [--strict]           profondeurs figees, environ 23 s\n"
        "  chess_perft deep      [--strict]           profondeur maximale publiee\n"
        "  chess_perft divide <fen|startpos> <n> [--strict] [--check-fen]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    PerftOptions opts;
    std::vector<std::string> positional;
    for (const std::string& arg : args) {
        if (arg == "--strict")         opts.strict = true;
        else if (arg == "--check-fen") opts.check_fen = true;
        else                           positional.push_back(arg);
    }

    if (positional.empty()) {
        print_usage();
        return 2;
    }

    const std::string& command = positional[0];

    if (command == "bench") {
        return run_campaign(3, opts, "Palier rapide (profondeur 1 a 3)");
    }
    if (command == "roundtrip") {
        PerftOptions controle = opts;
        controle.strict = true;
        controle.roundtrip = true;
        return run_campaign(3, controle,
                            "Roundtrip encodage / decodage (profondeur 1 a 3)");
    }
    if (command == "calibre") {
        return run_campaign(0, opts, "Palier calibre (profondeurs figees)",
                            true);
    }
    if (command == "deep") {
        return run_campaign(6, opts, "Palier profond");
    }

    if (command == "divide") {
        if (positional.size() < 3) {
            print_usage();
            return 2;
        }
        return cmd_divide(positional[1], std::stoi(positional[2]), opts);
    }

    print_usage();
    return 2;
}
