#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "chessboard.hpp"
#include "pgn_parser.hpp"

namespace fs = std::filesystem;

namespace {

void usage(const char* programme) {
    std::cout
        << "Usage: " << programme << " [tests...]\n"
        << "  --fen-file <fichier>  charge et verifie chaque FEN\n"
        << "  --pgn <fichier|dossier>  rejoue les PGN avec le parseur C++\n";
}

bool verifier_roundtrip(const Chessboard& board, const std::string& contexte) {
    Chessboard copie;
    try {
        copie.loadFEN(board.toFEN());
    } catch (const std::exception& exc) {
        std::cerr << "Echec du roundtrip " << contexte << " : "
                  << exc.what() << '\n';
        return false;
    }
    if (copie.getZobristHash() != board.getZobristHash()) {
        std::cerr << "Hash different apres roundtrip " << contexte << "\n"
                  << "avant=" << board.toFEN() << "\n"
                  << "apres=" << copie.toFEN() << '\n';
        return false;
    }
    return true;
}

bool tester_fens(const fs::path& chemin) {
    std::ifstream fichier(chemin);
    if (!fichier) {
        std::cerr << "Impossible d'ouvrir le fichier FEN : " << chemin << '\n';
        return false;
    }

    Chessboard board;
    std::string fen;
    size_t ligne = 0;
    size_t chargees = 0;
    const auto debut = std::chrono::steady_clock::now();
    while (std::getline(fichier, fen)) {
        ++ligne;
        if (fen.empty()) continue;
        try {
            board.loadFEN(fen);
        } catch (const std::exception& exc) {
            std::cerr << "FEN invalide ligne " << ligne << " : "
                      << exc.what() << '\n';
            return false;
        }
        if (!verifier_roundtrip(board, "FEN ligne " + std::to_string(ligne))) {
            return false;
        }
        ++chargees;
    }

    if (chargees == 0) {
        std::cerr << "Le fichier FEN est vide : " << chemin << '\n';
        return false;
    }
    const double secondes = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - debut).count();
    std::cout << "FEN : " << chargees << " positions verifiees en "
              << secondes << " s\n";
    return true;
}

std::vector<fs::path> fichiers_pgn(const fs::path& chemin) {
    if (fs::is_regular_file(chemin)) return {chemin};
    if (!fs::is_directory(chemin)) return {};

    std::vector<fs::path> resultats;
    for (const auto& entree : fs::directory_iterator(chemin)) {
        if (entree.is_regular_file() && entree.path().extension() == ".pgn") {
            resultats.push_back(entree.path());
        }
    }
    return resultats;
}

bool tester_pgn(const fs::path& chemin) {
    const std::vector<fs::path> fichiers = fichiers_pgn(chemin);
    if (fichiers.empty()) {
        std::cerr << "Aucun fichier PGN trouve : " << chemin << '\n';
        return false;
    }

    size_t parties = 0;
    size_t plies = 0;
    for (const fs::path& fichier : fichiers) {
        PgnParser parser;
        if (!parser.parseFiles(fichier.string())) {
            std::cerr << "Lecture PGN impossible : " << fichier << '\n';
            return false;
        }

        Chessboard board;
        board.setStartupPieces();
        const std::vector<std::string> coups = parser.extractMoves();
        for (size_t i = 0; i < coups.size(); ++i) {
            if (!board.movePieceSAN(coups[i])) {
                std::cerr << "Coup SAN refuse dans " << fichier.filename()
                          << ", ply " << (i + 1) << " : " << coups[i] << '\n';
                return false;
            }
            ++plies;
        }
        if (!verifier_roundtrip(board, fichier.filename().string())) return false;
        ++parties;
    }

    std::cout << "PGN : " << parties << " parties et " << plies
              << " demi-coups verifies\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        usage(argv[0]);
        return 2;
    }

    bool succes = true;
    bool test_lance = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            return 0;
        }
        if ((argument == "--fen-file" || argument == "--pgn")
            && i + 1 < argc) {
            const fs::path chemin = argv[++i];
            test_lance = true;
            succes = (argument == "--fen-file" ? tester_fens(chemin)
                                                 : tester_pgn(chemin))
                     && succes;
            continue;
        }

        std::cerr << "Argument inconnu ou valeur manquante : " << argument << '\n';
        usage(argv[0]);
        return 2;
    }

    return test_lance && succes ? 0 : 1;
}
