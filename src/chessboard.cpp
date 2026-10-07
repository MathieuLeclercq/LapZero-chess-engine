#include <array>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cstdint>
#include <utility>
#include <cmath>
#include <iostream>
#include <ios>

#include "chessboard.hpp"
#include "piece.hpp"
#include "square.hpp"
#include "zobrist.hpp"
#include "move.hpp"

//...............Constructors...............

Chessboard::Chessboard()
{
    // setup an empty chessboard
    clear();
}

//...............Getters...............

const Square& Chessboard::getSquare(int file, int rank) const
{
    return m_board[rank * 8 + file];
}

Square& Chessboard::getSquare(int file, int rank)
{
    return m_board[rank * 8 + file];
}

int Chessboard::getNumberOfOccupiedSquares() const
{
    int count = 0;
    for (const auto& square : m_board)
    {
        if (square.checkOccupied())
        {
            count++;
        }
    }
    return count;
}

//void Chessboard::print() const
//{
//    for (int i = 7; i > -1; i--)
//    {
//        for (int j = 0; j < 8; j++)
//        {
//            std::cout << m_board[i * 8 + j].getPiece().getValue() << " ";
//        }
//        std::cout << std::endl;
//    }
//}

void Chessboard::print() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int rank = 7; rank >= 0; --rank) {
        std::cout << rank + 1 << " |"; // Affichage du numéro de ligne (1-8)

        for (int file = 0; file < 8; ++file) {
            Piece p = m_board[rank * 8 + file].getPiece();
            char c = ' ';

            if (p.getType() == NONE) {
                c = '.';
            }
            else {
                switch (p.getType()) {
                case PAWN:   c = 'P'; break;
                case KNIGHT: c = 'N'; break;
                case BISHOP: c = 'B'; break;
                case ROOK:   c = 'R'; break;
                case QUEEN:  c = 'Q'; break;
                case KING:   c = 'K'; break;
                default:     c = '?'; break;
                }
                // Minuscules pour les noirs, majuscules pour les blancs
                if (p.getColor() == BLACK) c = std::tolower(c);
            }
            std::cout << " " << c << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";

    // Affichage des états internes invisibles sur l'échiquier
    std::cout << "Trait (Turn)       : " << (m_turn == WHITE ? "Blanc (w)" : "Noir (b)") << "\n";

    std::cout << "Droits de Roque    : ";
    bool has_castling = false;
    if (m_short_castle_white) { std::cout << "K"; has_castling = true; }
    if (m_long_castle_white) { std::cout << "Q"; has_castling = true; }
    if (m_short_castle_black) { std::cout << "k"; has_castling = true; }
    if (m_long_castle_black) { std::cout << "q"; has_castling = true; }
    if (!has_castling) std::cout << "-";
    std::cout << "\n";

    std::cout << "En Passant         : ";
    if (m_en_passant) {
        char ep_file_char = 'a' + m_en_passant_file;
        std::cout << ep_file_char << "\n";
    }
    else {
        std::cout << "-\n";
    }

    std::cout << "Regle des 50 coups : " << m_half_move_clock << " demi-coups\n";
    std::cout << "Zobrist Hash       : " << std::hex << m_current_zobrist_hash << std::dec << "\n";
    std::cout << "---------------------------------------\n";
}

void Chessboard::print(std::array<Square, 64> some_board) const
{
    for (int i = 7; i > -1; i--)
    {
        for (int j = 0; j < 8; j++)
        {
            std::cout << some_board[i * 8 + j].getPiece().getType() << " ";
        }
        std::cout << std::endl;
    }
}

bool Chessboard::checkThreefoldRepetition() const {
    int count = 1;

    // On parcourt l'historique à l'envers (du snapshot le plus récent au plus ancien)
    for (auto it = m_snapshotHistory.rbegin(); it != m_snapshotHistory.rend(); ++it) {

        if (it->zobrist_hash == m_current_zobrist_hash) {
            count++;
            if (count >= 3) {
                return true; // Early exit : on a trouvé 3 occurrences, inutile de continuer
            }
        }

        // Si ce snapshot résulte d'un coup irréversible (capture ou poussée de pion),
        // toute position antérieure est fondamentalement différente. On coupe la recherche.
        if (it->half_move_clock == 0) {
            break;
        }
    }

    return false;
}

bool Chessboard::checkInsufficientMaterial() const
{

    // 1. Comptage rapide sans allocation de vecteur
    int wCount = 0;
    int bCount = 0;
    for (int i = 0; i < 64; ++i) {
        if (m_board[i].checkOccupied()) {
            if (m_board[i].getPiece().getColor() == WHITE) wCount++;
            else bCount++;
        }
    }

    // 2. Fast Path : Si trop de pièces, ce n'est pas une nulle
    if (wCount > 2 || bCount > 2) return false;
    if (wCount + bCount > 4) return false; // Optionnel : K+B vs K+B est le max (4 pièces)

    // 3. Si on arrive ici, on a peu de pièces, on fait l'analyse détaillée
    std::vector<std::pair<PieceType, int>> whitePieces;
    std::vector<std::pair<PieceType, int>> blackPieces;

    for (int i = 0; i < 64; ++i) {
        const Piece& p = m_board[i].getPiece();
        if (p.getType() == NONE) continue;

        // On stocke le type et l'indice de case (pour la couleur des fous)
        if (p.getColor() == WHITE) whitePieces.push_back({ p.getType(), i });
        else blackPieces.push_back({ p.getType(), i });
    }

    auto isMajorOrPawn = [](const std::vector<std::pair<PieceType, int>>& pieces) {
        for (auto const& [type, idx] : pieces) {
            if (type == PAWN || type == ROOK || type == QUEEN) return true;
        }
        return false;
        };

    // Si un camp a un Pion, une Tour ou une Dame, le mat est encore possible.
    if (isMajorOrPawn(whitePieces) || isMajorOrPawn(blackPieces)) return false;


    // 1. K vs K
    if (wCount == 1 && bCount == 1) return true;

    // 2. K+B vs K ou K+N vs K (et inversement)
    if ((wCount == 2 && bCount == 1) || (wCount == 1 && bCount == 2)) {
        const auto& sideWithTwo = (wCount == 2) ? whitePieces : blackPieces;
        for (auto const& [type, idx] : sideWithTwo) {
            if (type == BISHOP || type == KNIGHT) return true;
        }
    }

    // 3. K+B vs K+B (Fous sur même couleur)
    if (wCount == 2 && bCount == 2) {
        int whiteBishopSq = -1;
        int blackBishopSq = -1;

        for (auto const& [type, idx] : whitePieces)
            if (type == BISHOP) whiteBishopSq = idx;
        for (auto const& [type, idx] : blackPieces)
            if (type == BISHOP) blackBishopSq = idx;

        if (whiteBishopSq != -1 && blackBishopSq != -1) {
            auto isLight = [](int idx) { return ((idx / 8) + (idx % 8)) % 2 != 0; };
            if (isLight(whiteBishopSq) == isLight(blackBishopSq)) return true;
        }
    }

    return false;
}

const std::vector<Move>& Chessboard::getMoveHistory() const
{
    return m_moveHistory;
}

std::vector<Move>& Chessboard::getMoveHistory()
{
    return m_moveHistory;
}

const std::vector<std::array<Square, 64>>& Chessboard::getBoardHistory() const
{
    return m_boardHistory;
}

std::vector<std::array<Square, 64>>& Chessboard::getBoardHistory()
{
    return m_boardHistory;
}

int Chessboard::getInitialPlyOffset() const 
{ 
    return m_initial_ply_offset; 
}

int Chessboard::getHalfMoveClock() const
{
    return m_half_move_clock;
}

Color Chessboard::getTurn() const
{
    return m_turn;
}

GameState Chessboard::getGameState() const
{
    return m_current_state;
}

bool Chessboard::isInCheck() const
{
    // On part du roi et on regarde si des pièces le menacent
    Color color = m_turn;
    Color oppositeColor = (color == WHITE) ? BLACK : WHITE;

    // 1. Récupérer les coordonnées de notre roi
    int king_file = (color == WHITE) ? m_white_king_file : m_black_king_file;
    int king_rank = (color == WHITE) ? m_white_king_rank : m_black_king_rank;


    // 2. Vérifier les menaces de Pions
    int pawn_direction = (color == WHITE) ? 1 : -1;
    int pr = king_rank + pawn_direction;
    if (pr >= 0 && pr < 8)
    {
        if (king_file - 1 >= 0)
        {
            const Piece& p = m_board[pr * 8 + (king_file - 1)].getPiece();
            if (p.getType() == PAWN && p.getColor() == oppositeColor) return true;
        }
        if (king_file + 1 < 8)
        {
            const Piece& p = m_board[pr * 8 + (king_file + 1)].getPiece();
            if (p.getType() == PAWN && p.getColor() == oppositeColor) return true;
        }
    }

    // 3. Vérifier les menaces de Cavaliers
    static constexpr int knight_moves[8][2] = { {1, 2}, {2, 1}, {2, -1}, {1, -2}, 
                                                {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2} };
    for (int i = 0; i < 8; i++)
    {
        int r = king_rank + knight_moves[i][0];
        int f = king_file + knight_moves[i][1];
        if (r >= 0 && r < 8 && f >= 0 && f < 8)
        {
            const Piece& p = m_board[r * 8 + f].getPiece();
            if (p.getType() == KNIGHT && p.getColor() == oppositeColor)
                return true;
        }
    }

    // 4. Vérifier les menaces Lignes/Colonnes
    static constexpr int orth_dirs[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
    for (int d = 0; d < 4; d++)
    {
        for (int i = 1; i < 8; i++)
        {
            int r = king_rank + orth_dirs[d][0] * i;
            int f = king_file + orth_dirs[d][1] * i;
            if (r < 0 || r >= 8 || f < 0 || f >= 8) break;

            const Piece& p = m_board[r * 8 + f].getPiece();
            if (p.getType() != NONE)
            {
                if (p.getColor() == oppositeColor && (p.getType() == ROOK || p.getType() == QUEEN))
                    return true;
                break;
            }
        }
    }

    // 5. Vérifier les menaces Diagonales
    static constexpr int diag_dirs[4][2] = { {1, 1}, {1, -1}, {-1, -1}, {-1, 1} };
    for (int d = 0; d < 4; d++)
    {
        for (int i = 1; i < 8; i++)
        {
            int r = king_rank + diag_dirs[d][0] * i;
            int f = king_file + diag_dirs[d][1] * i;
            if (r < 0 || r >= 8 || f < 0 || f >= 8) break;

            const Piece& p = m_board[r * 8 + f].getPiece();
            if (p.getType() != NONE)
            {
                if (p.getColor() == oppositeColor && (p.getType() == BISHOP || p.getType() == QUEEN))
                    return true;
                break;
            }
        }
    }

    // 6. Vérifier le Roi adverse
    static constexpr int king_moves[8][2] = { {1, 0}, {1, 1}, {0, 1}, {-1, 1}, 
                                              {-1, 0}, {-1, -1}, {0, -1}, {1, -1} };
    for (int i = 0; i < 8; i++)
    {
        int r = king_rank + king_moves[i][0];
        int f = king_file + king_moves[i][1];
        if (r >= 0 && r < 8 && f >= 0 && f < 8)
        {
            const Piece& p = m_board[r * 8 + f].getPiece();
            if (p.getType() == KING && p.getColor() == oppositeColor)
                return true;
        }
    }

    return false;
}

void Chessboard::getLegalMovesForSquare(int file, int rank, std::vector<Move>& result,
    std::vector<Move>& pseudo_buffer,
    int filter_dest_file, int filter_dest_rank) {

    pseudo_buffer.clear(); // O(1), remet la taille à 0 mais garde la capacité (capacity)

    // Remplissage du buffer par la fonction naïve, sans nouvelle allocation
    getNaiveLegalMoves(file, rank, pseudo_buffer);

    PieceType p_type = m_board[rank * 8 + file].getPiece().getType();
    bool is_king_move = (p_type == KING);
    bool is_pawn = (p_type == PAWN);

    for (const Move& move : pseudo_buffer) {
        int dest_file = move.getDestSquare().getFile();
        int dest_rank = move.getDestSquare().getRank();

        if (filter_dest_file != -1 && (dest_file != filter_dest_file || dest_rank != filter_dest_rank))
            continue;

        if (is_king_move && std::abs(file - dest_file) == 2) {
            if (!isCastleSafe(file, rank, dest_file, dest_rank))
                continue;
        }

        bool is_en_passant = (is_pawn && std::abs(file - dest_file) == 1 &&
            m_board[dest_rank * 8 + dest_file].getPiece().getType() == NONE);

        if (isMoveSafe(file, rank, dest_file, dest_rank, is_en_passant, is_king_move)) {
            result.push_back(move);
        }
    }
}

std::vector<Move> Chessboard::getAllLegalMoves() {
    std::vector<Move> result;
    result.reserve(100); // pré-allocation de 100 coups possibles

    std::vector<Move> pseudo_buffer;
    // Une Dame au centre d'un plateau vide a 27 coups pseudo-légaux. 
    pseudo_buffer.reserve(27);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (m_board[j * 8 + i].getPiece().getColor() != m_turn)
                continue;

            // On fait descendre les deux vecteurs pré-alloués
            getLegalMovesForSquare(i, j, result, pseudo_buffer);
        }
    }
    return result;
}


bool Chessboard::hasAnyLegalMove() {
    // Buffer de travail unique pour toute la fonction
    std::vector<Move> pseudo_buffer;
    pseudo_buffer.reserve(27);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (m_board[j * 8 + i].getPiece().getColor() != m_turn)
                continue;

            pseudo_buffer.clear();
            getNaiveLegalMoves(i, j, pseudo_buffer);

            PieceType p_type = m_board[j * 8 + i].getPiece().getType();
            bool is_king_move = (p_type == KING);
            bool is_pawn = (p_type == PAWN);

            for (const Move& move : pseudo_buffer) {
                int dest_file = move.getDestSquare().getFile();
                int dest_rank = move.getDestSquare().getRank();

                if (is_king_move && std::abs(i - dest_file) == 2) {
                    if (!isCastleSafe(i, j, dest_file, dest_rank))
                        continue;
                }

                bool is_en_passant = (is_pawn && std::abs(i - dest_file) == 1 &&
                    m_board[dest_rank * 8 + dest_file].getPiece().getType() == NONE);

                if (isMoveSafe(i, j, dest_file, dest_rank, is_en_passant, is_king_move)) {
                    return true;  // early exit : un seul coup légal suffit
                }
            }
        }
    }
    return false;
}

Move Chessboard::decodeMoveIndex(int index) const
{
    const bool is_black = (getTurn() == BLACK);
    const int plane = index / 64;
    const int remainder = index % 64;
    int orig_r = remainder / 8;
    int orig_f = remainder % 8;

    int df = 0, dr = 0;
    PieceType promotion = NONE;

    if (plane < 56)
    {
        // Coups de dame : 8 directions x 7 distances.
        const int dir_idx = plane / 7;
        const int dist = (plane % 7) + 1;
        const int dirs[8][2] = { {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
        df = dirs[dir_idx][0] * dist;
        dr = dirs[dir_idx][1] * dist;
    }
    else if (plane < 64)
    {
        const int knight_idx = plane - 56;
        const int knight_moves[8][2] = { {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2} };
        df = knight_moves[knight_idx][0];
        dr = knight_moves[knight_idx][1];
    }
    else
    {
        // Sous-promotions : 3 directions x 3 pieces (cavalier, fou, tour).
        const int sub_idx = plane - 64;
        const int dir_idx = sub_idx / 3;
        const int p_idx = sub_idx % 3;
        df = dir_idx - 1;
        dr = 1;

        if (p_idx == 0) promotion = KNIGHT;
        else if (p_idx == 1) promotion = BISHOP;
        else promotion = ROOK;
    }

    int dest_f = orig_f + df;
    int dest_r = orig_r + dr;

    if (is_black)
    {
        orig_r = 7 - orig_r;
        dest_r = 7 - dest_r;
    }

    const Square origine = getSquare(orig_f, orig_r);
    // Une poussee de pion sur la derniere rang ee est une promotion dame quand
    // l'index ne precise pas de piece, c'est-a-dire sur un plan de dame.
    if (origine.getPiece().getType() == PAWN)
    {
        if ((!is_black && dest_r == 7) || (is_black && dest_r == 0))
        {
            if (promotion == NONE) promotion = QUEEN;
        }
    }

    return Move(origine, Square(dest_f, dest_r), promotion);
}

int encodeMoveIndex(int orig_f, int orig_r, int dest_f, int dest_r,
                    PieceType promotion, bool is_black)
{
    if (is_black)
    {
        orig_r = 7 - orig_r;
        dest_r = 7 - dest_r;
    }

    int df = dest_f - orig_f;
    int dr = dest_r - orig_r;
    int plane = -1;

    if (promotion == KNIGHT || promotion == BISHOP || promotion == ROOK)
    {
        int dir_idx = df + 1;
        int p_idx = 0;
        if (promotion == KNIGHT) p_idx = 0;
        else if (promotion == BISHOP) p_idx = 1;
        else if (promotion == ROOK) p_idx = 2;
        plane = 64 + dir_idx * 3 + p_idx;
    }
    else if ((std::abs(df) == 2 && std::abs(dr) == 1) || (std::abs(df) == 1 && std::abs(dr) == 2))
    {
        int knight_moves[8][2] = { {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2} };
        for (int i = 0; i < 8; ++i)
        {
            if (knight_moves[i][0] == df && knight_moves[i][1] == dr)
            {
                plane = 56 + i;
                break;
            }
        }
    }
    else
    {
        int dirs[8][2] = { {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
        int dist = std::max(std::abs(df), std::abs(dr));
        if (dist == 0) return -1;

        int dir_f = df / dist;
        int dir_r = dr / dist;
        int dir_idx = -1;
        for (int i = 0; i < 8; ++i)
        {
            if (dirs[i][0] == dir_f && dirs[i][1] == dir_r)
            {
                dir_idx = i;
                break;
            }
        }
        if (dir_idx != -1)
        {
            plane = dir_idx * 7 + (dist - 1);
        }
    }

    if (plane == -1) return -1;
    return plane * 64 + orig_r * 8 + orig_f;
}

int Chessboard::encodeMove(const Move& move) const
{
    return encodeMoveIndex(
        move.getOrigSquare().getFile(), move.getOrigSquare().getRank(),
        move.getDestSquare().getFile(), move.getDestSquare().getRank(),
        move.getPromotion(), m_turn == BLACK);
}

std::vector<int> Chessboard::getLegalMoveIndices() {
    std::vector<Move> all_legal_moves = getAllLegalMoves();

    std::vector<int> indices;
    indices.reserve(all_legal_moves.size());

    for (const Move& move : all_legal_moves)
    {
        int idx = encodeMove(move);
        if (idx != -1)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

void Chessboard::checkEnPassant()
{
    Move lastMove = m_moveHistory.back();
    if (lastMove.getPiece().getType() == PAWN && 
        abs(lastMove.getDestSquare().getRank() - lastMove.getOrigSquare().getRank()) == 2)
    {
        m_en_passant = true;
        m_en_passant_file = lastMove.getDestSquare().getFile();
    }
    else
        m_en_passant = false;
}

void Chessboard::printPly() const
{
    std::cout << "ply " << m_boardHistory.size() << "." << std::endl;
    print();
    std::string color_str = (m_turn == WHITE) ? "White" : "Black";
    std::cout << color_str << " to move" << "\n\n" << std::endl;
    bool check = isInCheck();
    //if (check)
    //    std::cout << "Check!" << std::endl;
}

//...............Setters...............

void Chessboard::clear() {
    // 1. Réinitialisation du tableau de cases
    m_board = std::array<Square, 64>();
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            m_board[j * 8 + i].setPosition(i, j);
        }
    }

    // 2. Réinitialisation des états de la partie
    m_current_state = ONGOING;
    m_turn = WHITE;
    m_half_move_clock = 0;

    // 3. Réinitialisation des positions des rois
    m_white_king_file = -1;
    m_white_king_rank = -1;
    m_black_king_file = -1;
    m_black_king_rank = -1;

    // 4. Réinitialisation des drapeaux spéciaux
    m_en_passant = false;
    m_en_passant_file = -1;

    m_short_castle_white = false;
    m_long_castle_white = false;
    m_short_castle_black = false;
    m_long_castle_black = false;

    // 5. Vidage des historiques
    m_moveHistory.clear();
    m_boardHistory.clear();
    m_snapshotHistory.clear();
    m_initial_ply_offset = 0;

    // 6. Réinitialisation du Hash
    m_current_zobrist_hash = 0;
}

void Chessboard::setStartupPieces()
{
    m_white_king_file = 4;
    m_white_king_rank = 0;
    m_black_king_file = 4;
    m_black_king_rank = 7;

    m_short_castle_white = true;
    m_long_castle_white = true;
    m_short_castle_black = true;
    m_long_castle_black = true;

    for (int i = 0; i < 8; i++) // i = file index
    {
        for (int j = 0; j < 8; j++) // j = rank index
        {
            int file = i + 1;
            int rank = j + 1;

            // pawns
            if (rank == 2)
            {
                m_board[j * 8 + i].setPiece(Piece(WHITE, PAWN));
            }
            else if (rank == 7)
            {
                m_board[j * 8 + i].setPiece(Piece(BLACK, PAWN));
            }
            // rooks
            else if (rank == 1 && (file == 1 || file == 8))
            {
                m_board[j * 8 + i].setPiece(Piece(WHITE, ROOK));
            }
            else if (rank == 8 && (file == 1 || file == 8))
            {
                m_board[j * 8 + i].setPiece(Piece(BLACK, ROOK));
            }
            // knights
            else if (rank == 1 && (file == 2 || file == 7))
            {
                m_board[j * 8 + i].setPiece(Piece(WHITE, KNIGHT));
            }
            else if (rank == 8 && (file == 2 || file == 7))
            {
                m_board[j * 8 + i].setPiece(Piece(BLACK, KNIGHT));
            }
            // bishops
            else if (rank == 1 && (file == 3 || file == 6))
            {
                m_board[j * 8 + i].setPiece(Piece(WHITE, BISHOP));
            }
            else if (rank == 8 && (file == 3 || file == 6))
            {
                m_board[j * 8 + i].setPiece(Piece(BLACK, BISHOP));
            }
            // queen
            else if (rank == 1 && file == 4)
            {
                m_board[j * 8 + i].setPiece(Piece(WHITE, QUEEN));
            }
            else if (rank == 8 && file == 4)
            {
                m_board[j * 8 + i].setPiece(Piece(BLACK, QUEEN));
            }
            // king
            else if (rank == 1 && file == 5)
            {
                m_board[j * 8 + i].setPiece(Piece(WHITE, KING));
            }
            else if (rank == 8 && file == 5)
            {
                m_board[j * 8 + i].setPiece(Piece(BLACK, KING));
            }
            // empty squares
            else
            {
                m_board[j * 8 + i].setPiece(Piece());
            }
        }
    }
    m_boardHistory.push_back(m_board);
    computeInitialZobrist();
}

void Chessboard::setAmnesiaMode(bool amnesia) { 
    m_amnesia_mode = amnesia; 
}

void Chessboard::loadFEN(const std::string& fen) {
    clear();

    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, en_passant_str, half_move_str, full_move_str;
    iss >> board_str >> turn_str >> castling_str >> en_passant_str >> half_move_str >> full_move_str;

    // --- CHAMP 1 : Placement des pièces ---
    int file = 0;
    int rank = 7;

    for (char c : board_str) {
        if (c == '/') {
            rank--;
            file = 0;
        }
        else if (std::isdigit(c)) {
            file += (c - '0');
        }
        else {
            Color color = std::isupper(c) ? WHITE : BLACK;
            PieceType type = NONE;
            switch (std::tolower(c)) {
            case 'p': type = PAWN; break;
            case 'n': type = KNIGHT; break;
            case 'b': type = BISHOP; break;
            case 'r': type = ROOK; break;
            case 'q': type = QUEEN; break;
            case 'k': type = KING; break;
            }
            m_board[rank * 8 + file].setPiece(Piece(color, type));

            if (type == KING) {
                if (color == WHITE) {
                    m_white_king_file = file;
                    m_white_king_rank = rank;
                }
                else {
                    m_black_king_file = file;
                    m_black_king_rank = rank;
                }
            }
            file++;
        }
    }

    m_turn = (turn_str == "w") ? WHITE : BLACK;

    // --- CHAMP 3 : Droits de roque ---
    if (!castling_str.empty() && castling_str != "-") {
        for (char c : castling_str) {
            if (c == 'K') m_short_castle_white = true;
            if (c == 'Q') m_long_castle_white = true;
            if (c == 'k') m_short_castle_black = true;
            if (c == 'q') m_long_castle_black = true;
        }
    }

    // --- CHAMP 4 : En passant ---
    if (!en_passant_str.empty() && en_passant_str != "-") {
        m_en_passant = true;
        m_en_passant_file = en_passant_str[0] - 'a';
    }

    // --- CHAMP 5 : Règle des 50 coups ---
    if (!half_move_str.empty()) {
        try {
            m_half_move_clock = std::stoi(half_move_str);
        }
        catch (...) {
            m_half_move_clock = 0; // Sécurité si la chaîne est invalide
        }
    }

    // --- CHAMP 6 : Nombre de plies joués --  
    if (!full_move_str.empty()) {
        try {
            int full_move = std::stoi(full_move_str);
            m_initial_ply_offset = (full_move - 1) * 2 + (m_turn == BLACK ? 1 : 0);
        }
        catch (...) {
            m_initial_ply_offset = 0;
        }
    }

    m_boardHistory.push_back(m_board);
    computeInitialZobrist();
}

std::string Chessboard::toFEN() const {
    std::string fen;

    // --- CHAMP 1 : Placement des pièces, de la rangée 8 vers la rangée 1 ---
    for (int rank = 7; rank >= 0; --rank) {
        int empty_run = 0;
        for (int file = 0; file < 8; ++file) {
            const Piece& p = m_board[rank * 8 + file].getPiece();
            if (p.getType() == NONE) {
                ++empty_run;
                continue;
            }
            if (empty_run > 0) {
                fen += std::to_string(empty_run);
                empty_run = 0;
            }
            char c = '?';
            switch (p.getType()) {
            case PAWN:   c = 'p'; break;
            case KNIGHT: c = 'n'; break;
            case BISHOP: c = 'b'; break;
            case ROOK:   c = 'r'; break;
            case QUEEN:  c = 'q'; break;
            case KING:   c = 'k'; break;
            default: break;
            }
            fen += (p.getColor() == WHITE)
                ? static_cast<char>(std::toupper(static_cast<unsigned char>(c)))
                : c;
        }
        if (empty_run > 0) fen += std::to_string(empty_run);
        if (rank > 0) fen += '/';
    }

    // --- CHAMP 2 : Trait ---
    fen += (m_turn == WHITE) ? " w " : " b ";

    // --- CHAMP 3 : Droits de roque ---
    std::string castling;
    if (m_short_castle_white) castling += 'K';
    if (m_long_castle_white)  castling += 'Q';
    if (m_short_castle_black) castling += 'k';
    if (m_long_castle_black)  castling += 'q';
    fen += castling.empty() ? "-" : castling;

    // --- CHAMP 4 : Case en passant ---
    // La FEN attend la case TRAVERSEE par le pion, pas sa case d'arrivée.
    // Si les Blancs ont le trait, les Noirs viennent de pousser de 7 vers 5,
    // la case traversée est donc sur la rangée 6. Symétriquement rangée 3.
    if (m_en_passant && m_en_passant_file >= 0 && m_en_passant_file < 8) {
        fen += ' ';
        fen += static_cast<char>('a' + m_en_passant_file);
        fen += (m_turn == WHITE) ? '6' : '3';
    }
    else {
        fen += " -";
    }

    // --- CHAMP 5 : Compteur de la règle des 50 coups ---
    fen += ' ';
    fen += std::to_string(m_half_move_clock);

    // --- CHAMP 6 : Numéro de coup complet ---
    const int plies_played = m_boardHistory.empty()
        ? 0
        : static_cast<int>(m_boardHistory.size()) - 1;
    const int total_ply = m_initial_ply_offset + plies_played;
    fen += ' ';
    fen += std::to_string(total_ply / 2 + 1);

    return fen;
}

void Chessboard::setKiwipete() {

    ///
    /// Test pour voir si les hashes zobrist marchent bien.
    /// c'est une position connue pour faire du débug de moteur d'échecs
    ///
    
    // 1. Vider le plateau
    for (int i = 0; i < 64; i++) {
        m_board[i].setPiece(Piece());
    }

    // 2. Placer les pièces Blanches
    m_board[0 * 8 + 0].setPiece(Piece(WHITE, ROOK));   // a1
    m_board[0 * 8 + 4].setPiece(Piece(WHITE, KING));   // e1
    m_board[0 * 8 + 7].setPiece(Piece(WHITE, ROOK));   // h1
    m_board[1 * 8 + 0].setPiece(Piece(WHITE, PAWN));   // a2
    m_board[1 * 8 + 1].setPiece(Piece(WHITE, PAWN));   // b2
    m_board[1 * 8 + 2].setPiece(Piece(WHITE, PAWN));   // c2
    m_board[1 * 8 + 3].setPiece(Piece(WHITE, BISHOP)); // d2
    m_board[1 * 8 + 4].setPiece(Piece(WHITE, BISHOP)); // e2
    m_board[1 * 8 + 5].setPiece(Piece(WHITE, PAWN));   // f2
    m_board[1 * 8 + 6].setPiece(Piece(WHITE, PAWN));   // g2
    m_board[1 * 8 + 7].setPiece(Piece(WHITE, PAWN));   // h2
    m_board[2 * 8 + 2].setPiece(Piece(WHITE, KNIGHT)); // c3
    m_board[2 * 8 + 5].setPiece(Piece(WHITE, QUEEN));  // f3
    m_board[3 * 8 + 4].setPiece(Piece(WHITE, PAWN));   // e4
    m_board[4 * 8 + 3].setPiece(Piece(WHITE, PAWN));   // d5
    m_board[4 * 8 + 4].setPiece(Piece(WHITE, KNIGHT)); // e5

    // 3. Placer les pièces Noires
    m_board[7 * 8 + 0].setPiece(Piece(BLACK, ROOK));   // a8
    m_board[7 * 8 + 4].setPiece(Piece(BLACK, KING));   // e8
    m_board[7 * 8 + 7].setPiece(Piece(BLACK, ROOK));   // h8
    m_board[6 * 8 + 0].setPiece(Piece(BLACK, PAWN));   // a7
    m_board[6 * 8 + 2].setPiece(Piece(BLACK, PAWN));   // c7
    m_board[6 * 8 + 3].setPiece(Piece(BLACK, PAWN));   // d7
    m_board[6 * 8 + 4].setPiece(Piece(BLACK, QUEEN));  // e7
    m_board[6 * 8 + 5].setPiece(Piece(BLACK, PAWN));   // f7
    m_board[6 * 8 + 6].setPiece(Piece(BLACK, BISHOP)); // g7
    m_board[5 * 8 + 0].setPiece(Piece(BLACK, BISHOP)); // a6
    m_board[5 * 8 + 1].setPiece(Piece(BLACK, KNIGHT)); // b6
    m_board[5 * 8 + 4].setPiece(Piece(BLACK, PAWN));   // e6
    m_board[5 * 8 + 5].setPiece(Piece(BLACK, KNIGHT)); // f6
    m_board[5 * 8 + 6].setPiece(Piece(BLACK, PAWN));   // g6
    m_board[3 * 8 + 1].setPiece(Piece(BLACK, PAWN));   // b4
    m_board[2 * 8 + 7].setPiece(Piece(BLACK, PAWN));   // h3

    // 4. Initialiser les métadonnées pour autoriser tous les roques
    m_turn = WHITE;
    m_short_castle_white = true;
    m_long_castle_white = true;
    m_short_castle_black = true;
    m_long_castle_black = true;
    m_en_passant = false;
    m_half_move_clock = 0;
    m_white_king_file = 4;
    m_white_king_rank = 0;
    m_black_king_file = 4;
    m_black_king_rank = 7;

    m_boardHistory.clear();
    m_boardHistory.push_back(m_board);
    m_moveHistory.clear();
    m_snapshotHistory.clear();

    // 5. Générer le hash initial
    computeInitialZobrist();
}

void Chessboard::setBoard(std::array<Square, 64> some_board)
{
    m_board = some_board;
    computeInitialZobrist();
}

void Chessboard::updateHistory(const Move& move)
{
    m_moveHistory.push_back(move);
    m_boardHistory.push_back(m_board);
}

void Chessboard::updateCastleFlags()
{
    // 3 cas à gérer : 
    // roi a bougé
    // tour a bougé
    // tour capturée (sans forcément avoir bougé avant)

    Move lastMove = m_moveHistory.back();

    // 1. Perte des deux droits si le roi bouge
    if (lastMove.getPiece().getType() == KING)
    {
        if (lastMove.getPiece().getColor() == WHITE)
        {
            m_short_castle_white = false;
            m_long_castle_white = false;
        }
        else if (lastMove.getPiece().getColor() == BLACK)
        {
            m_short_castle_black = false;
            m_long_castle_black = false;
        }
    }

    // 2. Perte d'un droit spécifique si une case de tour est impliquée (départ ou arrivée)
    int orig_f = lastMove.getOrigSquare().getFile();
    int orig_r = lastMove.getOrigSquare().getRank();
    int dest_f = lastMove.getDestSquare().getFile();
    int dest_r = lastMove.getDestSquare().getRank();

    // Tour blanche a1 (Grand roque blanc)
    if ((orig_f == 0 && orig_r == 0) || (dest_f == 0 && dest_r == 0))
        m_long_castle_white = false;

    // Tour blanche h1 (Petit roque blanc)
    if ((orig_f == 7 && orig_r == 0) || (dest_f == 7 && dest_r == 0))
        m_short_castle_white = false;

    // Tour noire a8 (Grand roque noir)
    if ((orig_f == 0 && orig_r == 7) || (dest_f == 0 && dest_r == 7))
        m_long_castle_black = false;

    // Tour noire h8 (Petit roque noir)
    if ((orig_f == 7 && orig_r == 7) || (dest_f == 7 && dest_r == 7))
        m_short_castle_black = false;
}



void Chessboard::applyPromotion(Square& second_square, PieceType force_promotion)
{
    if (force_promotion != NONE)
    {
        second_square.setPiece(Piece(second_square.getPiece().getColor(), force_promotion));
    }
}

bool Chessboard::isCastleSafe(int orig_file, int orig_rank, int file, int rank)
{
    bool short_castle = (file == 6);
    bool long_castle = (file == 2);

    if (!short_castle && !long_castle) return false;

    int dir = short_castle ? 1 : -1;

    // Vérifie que les 3 cases traversées par le roi (départ, milieu, arrivée) ne sont pas contrôlées
    for (int i = 0; i <= 2; i++)
    {
        int current_file = orig_file + (i * dir);
        if (!isMoveSafe(orig_file, orig_rank, current_file, rank, false, true))
        {
            return false;
        }
    }

    return true;
}

bool Chessboard::isMoveSafe(int orig_f, int orig_r,
    int dest_f, int dest_r,
    bool is_en_passant, bool is_king_move) {
    Square& orig_sq = m_board[orig_r * 8 + orig_f];
    Square& dest_sq = m_board[dest_r * 8 + dest_f];

    Piece moving_piece = orig_sq.getPiece();
    Piece captured_piece = dest_sq.getPiece();
    Piece ep_captured_piece;

    int ep_file = dest_f;
    int ep_rank = orig_r; // Le pion capturé en passant est sur la même rangée que le pion d'origine

    // 1. Sauvegarde locale et modification
    if (is_en_passant) {
        ep_captured_piece = m_board[ep_rank * 8 + ep_file].getPiece();
        m_board[ep_rank * 8 + ep_file].setPiece(Piece());
    }

    // Application du coup
    dest_sq.setPiece(moving_piece);
    orig_sq.setPiece(Piece());

    int old_king_f = -1, old_king_r = -1;
    if (is_king_move) {
        if (m_turn == WHITE) {
            old_king_f = m_white_king_file; old_king_r = m_white_king_rank;
            m_white_king_file = dest_f; m_white_king_rank = dest_r;
        }
        else {
            old_king_f = m_black_king_file; old_king_r = m_black_king_rank;
            m_black_king_file = dest_f; m_black_king_rank = dest_r;
        }
    }

    // 2. Vérification
    bool in_check = isInCheck();

    // 3. Restauration
    orig_sq.setPiece(moving_piece);
    dest_sq.setPiece(captured_piece);

    if (is_en_passant) {
        m_board[ep_rank * 8 + ep_file].setPiece(ep_captured_piece);
    }

    if (is_king_move) {
        if (m_turn == WHITE) {
            m_white_king_file = old_king_f; m_white_king_rank = old_king_r;
        }
        else {
            m_black_king_file = old_king_f; m_black_king_rank = old_king_r;
        }
    }

    return !in_check;
}

void Chessboard::evaluateGameState()
{
    if (!hasAnyLegalMove())  // soit mat soit pat
    {
        if (isInCheck())
        {
            m_current_state = CHECKMATE;
        }
        else
        {
            m_current_state = STALEMATE;
        }
    }
    else if (checkThreefoldRepetition())
    {
        m_current_state = DRAW_REPETITION;
    }
    else if (m_half_move_clock >= 100) // Fin de partie par la règle des 50 coups
    {
        m_current_state = DRAW_50_MOVES;
    }

    else if (checkInsufficientMaterial())
    {
        m_current_state = DRAW_INSUFF_MATERIAL;
    }
}

void Chessboard::updateStateSnapshot()
{
    StateSnapshot current_snapshot;
    current_snapshot.short_castle_white = m_short_castle_white;
    current_snapshot.long_castle_white = m_long_castle_white;
    current_snapshot.short_castle_black = m_short_castle_black;
    current_snapshot.long_castle_black = m_long_castle_black;
    current_snapshot.en_passant = m_en_passant;
    current_snapshot.en_passant_file = m_en_passant_file;
    current_snapshot.half_move_clock = m_half_move_clock;
    current_snapshot.current_state = m_current_state;
    current_snapshot.white_king_file = m_white_king_file;
    current_snapshot.white_king_rank = m_white_king_rank;
    current_snapshot.black_king_file = m_black_king_file;
    current_snapshot.black_king_rank = m_black_king_rank;
    current_snapshot.zobrist_hash = m_current_zobrist_hash;
    m_snapshotHistory.push_back(current_snapshot);
}

bool Chessboard::movePiece(int orig_file, int orig_rank, 
                           int file, int rank, 
                           PieceType promotion, bool check_game_end)
{
    Square& first_square = m_board[orig_rank * 8 + orig_file];
    Square& second_square = m_board[rank * 8 + file];

    if (first_square.getPiece().getColor() != m_turn)
    {
        return false;
    }

    std::vector<Move> valid_moves;
    std::vector<Move> pseudo_buffer;
    pseudo_buffer.reserve(27); // 27 coups légaux max pour 1 pièce
    valid_moves.reserve(100);  // sauf cas extremement rare : 100 coups légaux largement assez
    getLegalMovesForSquare(orig_file, orig_rank, valid_moves, pseudo_buffer, file, rank);

    Move attempted_move(first_square, second_square, promotion);
    if (std::find(valid_moves.begin(), valid_moves.end(), attempted_move) == valid_moves.end())
    {
        return false; // si le move tenté n'est pas dans la liste des coups possibles
    }

    // A PARTIR D'ICI LE COUP EST VALIDE
    // 1. CAPTURE DE L'ÉTAT AVANT TOUTE MODIFICATION
    
    updateStateSnapshot();

    Piece moving_piece = first_square.getPiece();
    bool is_king_move = (moving_piece.getType() == KING);
    bool is_pawn_move = (moving_piece.getType() == PAWN);
    Color moving_color = moving_piece.getColor();

    bool is_en_passant_capture = (is_pawn_move && second_square.getPiece().getType() == NONE &&
        abs(orig_file - file) == 1);  // pion a bougé en diagonale sur une case vide
    bool is_capture = second_square.checkOccupied() || is_en_passant_capture;


    // --- ZOBRIST (Partie 1) : Retrait de l'ancien état ---
    m_current_zobrist_hash ^= Zobrist::BLACK_TO_MOVE; // On change le trait

    // On retire les anciens droits de roque et en_passant
    int old_castling_idx = (m_short_castle_white ? 1 : 0) | 
                           (m_long_castle_white ? 2 : 0)  | 
                           (m_short_castle_black ? 4 : 0) | 
                           (m_long_castle_black ? 8 : 0);
    m_current_zobrist_hash ^= Zobrist::CASTLING_KEYS[old_castling_idx];
    if (m_en_passant && m_en_passant_file >= 0) {
        m_current_zobrist_hash ^= Zobrist::EN_PASSANT_KEYS[m_en_passant_file];
    }

    // Retrait de la pièce de sa case de départ
    int orig_square_idx = orig_rank * 8 + orig_file;
    int dest_square_idx = rank * 8 + file;
    int moving_piece_idx = moving_piece.getZobristIndex();
    m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[orig_square_idx][moving_piece_idx];

    // Ajout à la case d'arrivée (Gestion de la promotion)
    if (promotion != NONE) {
        Piece promoted_piece(moving_color, promotion);
        m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[dest_square_idx][promoted_piece.getZobristIndex()];
    }
    else {
        m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[dest_square_idx][moving_piece_idx];
    }

    // Retrait de la pièce capturée
    if (is_capture) {
        if (is_en_passant_capture) {
            Piece captured_pawn(m_turn == WHITE ? BLACK : WHITE, PAWN);
            m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[orig_rank * 8 + file][captured_pawn.getZobristIndex()];
        }
        else {
            m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[dest_square_idx][second_square.getPiece().getZobristIndex()];
        }
    }

    // Gestion du Roque (Mouvement de la tour)
    if (is_king_move && abs(orig_file - file) == 2) {
        int rook_orig_file = (file > orig_file) ? 7 : 0;
        int rook_dest_file = (file > orig_file) ? 5 : 3;
        Piece rook(moving_color, ROOK);
        int rook_idx = rook.getZobristIndex();

        m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[rank * 8 + rook_orig_file][rook_idx]; // Retire la tour
        m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[rank * 8 + rook_dest_file][rook_idx]; // Replace la tour
    }


    // Exécution du coup

    // Roque
    if (is_king_move && abs(orig_file - file) == 2)
    {
        // déplacement de la tour
        int rook_orig_file = (file > orig_file) ? 7 : 0; // Tour h (7) pour petit roque, a (0) pour grand
        int rook_dest_file = (file > orig_file) ? 5 : 3; // Tour atterrit en f (5) ou d (3)

        m_board[rank * 8 + rook_dest_file].setPiece(m_board[rank * 8 + rook_orig_file].getPiece());
        m_board[rank * 8 + rook_orig_file].setPiece(Piece());
    }

    if (is_en_passant_capture)
    {
        m_board[orig_rank * 8 + file].setPiece(Piece()); // suppression du pion PAS SUR CASE D'ARRIVEE !!
    }

    // déplacement de la pièce qui a fait le coup actuel
    second_square.setPiece(first_square.getPiece());
    first_square.setPiece(Piece());

    // Mise à jour du cache si le roi bouge
    if (is_king_move)
    {
        if (moving_color == WHITE) {
            m_white_king_file = file; m_white_king_rank = rank;
        }
        else {
            m_black_king_file = file; m_black_king_rank = rank;
        }
    }
    

    // Préparation de l'état suivant
    applyPromotion(second_square, promotion);
    updateHistory(attempted_move);

    updateCastleFlags();
    checkEnPassant();


    // --- ZOBRIST (Partie 2) : Ajout des nouveaux droits ---
    int new_castling_idx = (m_short_castle_white ? 1 : 0) | 
        (m_long_castle_white ? 2 : 0) | 
        (m_short_castle_black ? 4 : 0) | 
        (m_long_castle_black ? 8 : 0);
    m_current_zobrist_hash ^= Zobrist::CASTLING_KEYS[new_castling_idx];
    if (m_en_passant && m_en_passant_file >= 0) {
        m_current_zobrist_hash ^= Zobrist::EN_PASSANT_KEYS[m_en_passant_file];
    }

    m_turn = (m_turn == WHITE) ? BLACK : WHITE;

    // maj du compteur de la règle des 50 coups
    if (is_capture || is_pawn_move) {
        m_half_move_clock = 0;
    }
    else {
        m_half_move_clock++;
    }

    if (check_game_end)
    {
        evaluateGameState();
    }

    return true;

}

bool Chessboard::movePiece(std::string orig_square, std::string square)
{
    int orig_file = orig_square[0] - 'a';
    int orig_rank = orig_square[1] - '1';
    int file = square[0] - 'a';
    int rank = square[1] - '1';

    return movePiece(orig_file, orig_rank, file, rank);
}

bool Chessboard::movePieceSAN(std::string san)
{
    // 1. Nettoyage de la chaîne
    san.erase(std::remove(san.begin(), san.end(), '+'), san.end());
    san.erase(std::remove(san.begin(), san.end(), '#'), san.end());
    san.erase(std::remove(san.begin(), san.end(), '!'), san.end());
    san.erase(std::remove(san.begin(), san.end(), '?'), san.end());

    // 2. Traitement des roques
    if (san == "O-O" || san == "O-O-O")
    {
        int rank = (m_turn == WHITE) ? 0 : 7;
        int orig_file = 4;
        int dest_file = (san == "O-O") ? 6 : 2;
        return movePiece(orig_file, rank, dest_file, rank);
    }

    // 3. Traitement des promotions
    PieceType promotion_type = NONE;
    size_t equal_pos = san.find('=');
    if (equal_pos != std::string::npos)
    {
        char p = san.back();
        if (p == 'Q') promotion_type = QUEEN;
        else if (p == 'R') promotion_type = ROOK;
        else if (p == 'B') promotion_type = BISHOP;
        else if (p == 'N') promotion_type = KNIGHT;
        san = san.substr(0, equal_pos);
    }

    if (san.length() < 2) return false;

    // 4. Extraction de la destination
    char dest_f_char = san[san.length() - 2];
    char dest_r_char = san[san.length() - 1];
    int dest_file = dest_f_char - 'a';
    int dest_rank = dest_r_char - '1';
    san = san.substr(0, san.length() - 2);

    // 5. Déduction du type de pièce
    PieceType p_type = PAWN;
    if (!san.empty() && isupper(san[0]))
    {
        char p = san[0];
        if (p == 'K') p_type = KING;
        else if (p == 'Q') p_type = QUEEN;
        else if (p == 'R') p_type = ROOK;
        else if (p == 'B') p_type = BISHOP;
        else if (p == 'N') p_type = KNIGHT;
        san = san.substr(1);
    }

    // 6. Suppression du symbole de capture
    if (!san.empty() && san.back() == 'x')
    {
        san.pop_back();
    }

    // 7. Extraction des indices de désambiguïsation
    int orig_file_hint = -1;
    int orig_rank_hint = -1;
    for (char c : san)
    {
        if (c >= 'a' && c <= 'h') orig_file_hint = c - 'a';
        if (c >= '1' && c <= '8') orig_rank_hint = c - '1';
    }

    // 8. Recherche du coup parmi les pièces du bon type uniquement
    int final_orig_file = -1;
    int final_orig_rank = -1;
    int match_count = 0;

    // Déclaration des buffers extérieurs aux boucles
    std::vector<Move> pseudo_buffer;
    pseudo_buffer.reserve(27);
    std::vector<Move> moves;
    moves.reserve(100);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            const Square& sq = m_board[j * 8 + i];
            if (!sq.checkOccupied() || sq.getPiece().getColor() != m_turn || sq.getPiece().getType() != p_type)
                continue;
            if (orig_file_hint != -1 && i != orig_file_hint) continue;
            if (orig_rank_hint != -1 && j != orig_rank_hint) continue;

            // On vide les vecteurs pour les réutiliser proprement
            moves.clear();

            // On fait descendre les deux vecteurs par référence
            getLegalMovesForSquare(i, j, moves, pseudo_buffer, dest_file, dest_rank);

            for (const Move& m : moves)
            {
                if (m.getDestSquare().getFile() != dest_file || m.getDestSquare().getRank() != dest_rank)
                    continue;
                if (promotion_type != NONE && m.getPromotion() != promotion_type)
                    continue;

                final_orig_file = i;
                final_orig_rank = j;
                match_count++;
            }
        }
    }

    if (match_count == 1)
    {
        return movePiece(final_orig_file, final_orig_rank, dest_file, dest_rank, promotion_type);
    }
    else
    {
        std::cerr << "Erreur SAN : " << match_count << " origines trouvées pour le coup " << san << std::endl;
        return false;
    }
}

bool Chessboard::movePieceUCI(const std::string& uci)
{
    if (uci.size() < 4 || uci.size() > 5) return false;

    const int orig_file = uci[0] - 'a';
    const int orig_rank = uci[1] - '1';
    const int dest_file = uci[2] - 'a';
    const int dest_rank = uci[3] - '1';

    if (orig_file < 0 || orig_file > 7 || orig_rank < 0 || orig_rank > 7 ||
        dest_file < 0 || dest_file > 7 || dest_rank < 0 || dest_rank > 7)
        return false;

    PieceType promotion = NONE;
    if (uci.size() == 5) {
        switch (std::tolower(static_cast<unsigned char>(uci[4]))) {
        case 'q': promotion = QUEEN;  break;
        case 'r': promotion = ROOK;   break;
        case 'b': promotion = BISHOP; break;
        case 'n': promotion = KNIGHT; break;
        default: return false;
        }
    }
    else if (m_board[orig_rank * 8 + orig_file].getPiece().getType() == PAWN
             && (dest_rank == 0 || dest_rank == 7)) {
        // Convention AlphaZero : promotion en dame par défaut.
        promotion = QUEEN;
    }

    return movePiece(orig_file, orig_rank, dest_file, dest_rank, promotion, false);
}

void Chessboard::undoMove()
{
    if (m_moveHistory.empty()) return;

    // 1. Retrait du dernier coup
    m_moveHistory.pop_back();
    m_boardHistory.pop_back();

    // La position précédente est maintenant le dernier élément de boardHistory
    // (car l'état initial est indexé en 0 par setStartupPieces)
    m_board = m_boardHistory.back();

    // 2. Restauration des métadonnées via le snapshot
    StateSnapshot snapshot = m_snapshotHistory.back();
    m_snapshotHistory.pop_back();

    m_short_castle_white = snapshot.short_castle_white;
    m_long_castle_white = snapshot.long_castle_white;
    m_short_castle_black = snapshot.short_castle_black;
    m_long_castle_black = snapshot.long_castle_black;
    m_en_passant = snapshot.en_passant;
    m_en_passant_file = snapshot.en_passant_file;
    m_half_move_clock = snapshot.half_move_clock;
    m_current_state = snapshot.current_state;
    m_white_king_file = snapshot.white_king_file;
    m_white_king_rank = snapshot.white_king_rank;
    m_black_king_file = snapshot.black_king_file;
    m_black_king_rank = snapshot.black_king_rank;
    m_current_zobrist_hash = snapshot.zobrist_hash;

    // 3. Restitution du trait
    m_turn = (m_turn == WHITE) ? BLACK : WHITE;
}

void Chessboard::getAlphaZeroTensor(std::vector<float>& tensor) const
{
    // Écrase les anciennes données avec des zéros sans faire de nouvelle allocation
    tensor.assign(119 * 64, 0.0f);

    Color p1_color = m_turn;
    Color p2_color = (p1_color == WHITE) ? BLACK : WHITE;
    bool flip = (p1_color == BLACK);

    // Construction d'un historique plat des Zobrist Hashs pour une vérification rapide
    std::vector<uint64_t> all_hashes;
    all_hashes.reserve(m_snapshotHistory.size() + 1);
    for (const auto& snap : m_snapshotHistory) {
        all_hashes.push_back(snap.zobrist_hash);
    }
    all_hashes.push_back(m_current_zobrist_hash);

    int history_depth = m_amnesia_mode ? 1 : 8;

    // Remplissage de l'historique (112 premiers plans)
    for (int t = 0; t < history_depth; t++)
    {
        int history_idx = m_boardHistory.size() - 1 - t;
        if (history_idx < 0) break;

        const std::array<Square, 64>& hist_board = m_boardHistory[history_idx];
        int plane_offset = t * 14 * 64;

        // --- CALCUL DES RÉPÉTITIONS VIA ZOBRIST ---
        uint64_t target_hash = all_hashes[history_idx];
        int rep_count = 0;
        for (int i = 0; i <= history_idx; i++) {
            if (all_hashes[i] == target_hash) {
                rep_count++;
            }
        }

        for (int rank = 0; rank < 8; rank++)
        {
            for (int file = 0; file < 8; file++)
            {
                const Piece& piece = hist_board[rank * 8 + file].getPiece();
                // Calcul des coordonnées inversées si c'est aux Noirs de jouer
                int tensor_rank = flip ? (7 - rank) : rank;
                int flat_idx = tensor_rank * 8 + file;

                if (piece.getType() != NONE)
                {
                    int piece_idx = -1;
                    switch (piece.getType()) {
                    case PAWN:   piece_idx = 0; break;
                    case KNIGHT: piece_idx = 1; break;
                    case BISHOP: piece_idx = 2; break;
                    case ROOK:   piece_idx = 3; break;
                    case QUEEN:  piece_idx = 4; break;
                    case KING:   piece_idx = 5; break;
                    default: break;
                    }

                    // P1 = 0 à 5, P2 = 6 à 11
                    if (piece.getColor() == p1_color) {
                        tensor[plane_offset + piece_idx * 64 + flat_idx] = 1.0f;
                    }
                    else {
                        tensor[plane_offset + (6 + piece_idx) * 64 + flat_idx] = 1.0f;
                    }
                }

                // Plans 12 et 13 : Répétitions
                if (rep_count == 2) {
                    tensor[plane_offset + 12 * 64 + flat_idx] = 1.0f;
                }
                else if (rep_count >= 3) {
                    tensor[plane_offset + 13 * 64 + flat_idx] = 1.0f;
                }
            }
        }
    }

    // Remplissage des 7 plans de contexte (Offset = 112 * 64 = 7168)
    int constant_offset = 112 * 64;

    float color_val = (m_turn == WHITE) ? 1.0f : 0.0f;
    // Normalisation des compteurs pour le réseau de neurones (sur une base arbitraire de 100)
    float total_moves_val = std::min(1.0f, (float)(m_boardHistory.size() / 2) / 100.0f);
    float p1_castle_k = (p1_color == WHITE) ? 
        (m_short_castle_white ? 1.0f : 0.0f) : (m_short_castle_black ? 1.0f : 0.0f);
    float p1_castle_q = (p1_color == WHITE) ? 
        (m_long_castle_white ? 1.0f : 0.0f) : (m_long_castle_black ? 1.0f : 0.0f);
    float p2_castle_k = (p2_color == WHITE) ? 
        (m_short_castle_white ? 1.0f : 0.0f) : (m_short_castle_black ? 1.0f : 0.0f);
    float p2_castle_q = (p2_color == WHITE) ? 
        (m_long_castle_white ? 1.0f : 0.0f) : (m_long_castle_black ? 1.0f : 0.0f);
    float no_progress_val = (float)m_half_move_clock / 100.0f;

    for (int i = 0; i < 64; i++)
    {
        tensor[constant_offset + 0 * 64 + i] = color_val;
        tensor[constant_offset + 1 * 64 + i] = total_moves_val;
        tensor[constant_offset + 2 * 64 + i] = p1_castle_k;
        tensor[constant_offset + 3 * 64 + i] = p1_castle_q;
        tensor[constant_offset + 4 * 64 + i] = p2_castle_k;
        tensor[constant_offset + 5 * 64 + i] = p2_castle_q;
        tensor[constant_offset + 6 * 64 + i] = no_progress_val;
    }
}


void Chessboard::getNaiveLegalMoves(int file, int rank, std::vector<Move>& pseudo_buffer) const
{
    // pour une case, retourne la liste des cases de déplacement dispos.
    // C'est un check naïf : il ne repère pas les clouages.
    // Il ne check pas non plus si le roque est safe

    Square orig_square = m_board[rank * 8 + file];
    std::vector<Move> legalMoves;
    PieceType type = orig_square.getPiece().getType();
    Color color = orig_square.getPiece().getColor();

    Color oppositeColor = (color == WHITE) ? BLACK : WHITE;

    // Fonction lambda pour gérer automatiquement les sous-promotions
    auto addMove = [&](const Square& target_square) {
        if (type == PAWN && (target_square.getRank() == 0 || target_square.getRank() == 7)) {
            pseudo_buffer.emplace_back(orig_square, target_square, QUEEN);
            pseudo_buffer.emplace_back(orig_square, target_square, ROOK);
            pseudo_buffer.emplace_back(orig_square, target_square, BISHOP);
            pseudo_buffer.emplace_back(orig_square, target_square, KNIGHT);
        }
        else {
            pseudo_buffer.emplace_back(orig_square, target_square, NONE);
        }
    };

    switch (type)
    {
    case PAWN:
    {
        if (color == WHITE && rank < 7 && m_board[(rank + 1) * 8 + file].checkOccupied() == false)
        {
            addMove(m_board[(rank + 1) * 8 + file]);
            if (rank == 1 && m_board[(rank + 2) * 8 + file].checkOccupied() == false)
            {
                addMove(m_board[(rank + 2) * 8 + file]);
            }
        }
        else if (color == BLACK && rank > 0 && m_board[(rank - 1) * 8 + file].checkOccupied() == false)
        {
            addMove(m_board[(rank - 1) * 8 + file]);
            if (rank == 6 && m_board[(rank - 2) * 8 + file].checkOccupied() == false)
            {
                addMove(m_board[(rank - 2) * 8 + file]);
            }
        }

        if (color == WHITE && rank < 7)
        {
            if (file < 7 && m_board[(rank + 1) * 8 + (file + 1)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank + 1) * 8 + (file + 1)]);
            }
            if (file > 0 && m_board[(rank + 1) * 8 + (file - 1)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank + 1) * 8 + (file - 1)]);
            }
            if (m_en_passant && rank == 4 && abs(file - m_en_passant_file) == 1)
            {
                addMove(m_board[(rank + 1) * 8 + m_en_passant_file]);
            }
        }
        else if (color == BLACK && rank > 0)
        {
            if (file < 7 && m_board[(rank - 1) * 8 + (file + 1)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank - 1) * 8 + (file + 1)]);
            }
            if (file > 0 && m_board[(rank - 1) * 8 + (file - 1)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank - 1) * 8 + (file - 1)]);
            }
            if (m_en_passant && rank == 3 && abs(file - m_en_passant_file) == 1)
            {
                addMove(m_board[(rank - 1) * 8 + m_en_passant_file]);
            }
        }
        break;
    }
    case ROOK:
    {
        for (int i = rank + 1; i < 8; i++) // top
        {
            if (m_board[i * 8 + file].checkOccupied() == false)
                addMove(m_board[i * 8 + file]);
            else if (m_board[i * 8 + file].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[i * 8 + file]);
                break;
            }
            else break;
        }
        for (int i = rank - 1; i > -1; i--) // bottom
        {
            if (m_board[i * 8 + file].checkOccupied() == false)
                addMove(m_board[i * 8 + file]);
            else if (m_board[i * 8 + file].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[i * 8 + file]);
                break;
            }
            else break;
        }
        for (int i = file + 1; i < 8; i++) // right
        {
            if (m_board[rank * 8 + i].checkOccupied() == false)
                addMove(m_board[rank * 8 + i]);
            else if (m_board[rank * 8 + i].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[rank * 8 + i]);
                break;
            }
            else break;
        }
        for (int i = file - 1; i > -1; i--) // left
        {
            if (m_board[rank * 8 + i].checkOccupied() == false)
                addMove(m_board[rank * 8 + i]);
            else if (m_board[rank * 8 + i].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[rank * 8 + i]);
                break;
            }
            else break;
        }
        break;
    }
    case KNIGHT:
    {
        if (file + 2 < 8 && rank + 1 < 8 && m_board[(rank + 1) * 8 + (file + 2)].getPiece().getColor() != color)
            addMove(m_board[(rank + 1) * 8 + (file + 2)]);
        if (file + 2 < 8 && rank - 1 > -1 && m_board[(rank - 1) * 8 + (file + 2)].getPiece().getColor() != color)
            addMove(m_board[(rank - 1) * 8 + (file + 2)]);
        if (file - 2 > -1 && rank + 1 < 8 && m_board[(rank + 1) * 8 + (file - 2)].getPiece().getColor() != color)
            addMove(m_board[(rank + 1) * 8 + (file - 2)]);
        if (file - 2 > -1 && rank - 1 > -1 && m_board[(rank - 1) * 8 + (file - 2)].getPiece().getColor() != color)
            addMove(m_board[(rank - 1) * 8 + (file - 2)]);
        if (file + 1 < 8 && rank + 2 < 8 && m_board[(rank + 2) * 8 + (file + 1)].getPiece().getColor() != color)
            addMove(m_board[(rank + 2) * 8 + (file + 1)]);
        if (file + 1 < 8 && rank - 2 > -1 && m_board[(rank - 2) * 8 + (file + 1)].getPiece().getColor() != color)
            addMove(m_board[(rank - 2) * 8 + (file + 1)]);
        if (file - 1 > -1 && rank + 2 < 8 && m_board[(rank + 2) * 8 + (file - 1)].getPiece().getColor() != color)
            addMove(m_board[(rank + 2) * 8 + (file - 1)]);
        if (file - 1 > -1 && rank - 2 > -1 && m_board[(rank - 2) * 8 + (file - 1)].getPiece().getColor() != color)
            addMove(m_board[(rank - 2) * 8 + (file - 1)]);
        break;
    }
    case BISHOP:
    {
        for (int i = 1; i < 8; i++) // up right
        {
            if (file + i < 8 && rank + i < 8 && m_board[(rank + i) * 8 + (file + i)].checkOccupied() == false)
                addMove(m_board[(rank + i) * 8 + (file + i)]);
            else if (file + i < 8 && rank + i < 8 && 
                     m_board[(rank + i) * 8 + (file + i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank + i) * 8 + (file + i)]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // up left
        {
            if (file - i > -1 && rank + i < 8 && m_board[(rank + i) * 8 + (file - i)].checkOccupied() == false)
                addMove(m_board[(rank + i) * 8 + (file - i)]);
            else if (file - i > -1 && rank + i < 8 && 
                     m_board[(rank + i) * 8 + (file - i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank + i) * 8 + (file - i)]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // down right
        {
            if (file + i < 8 && rank - i > -1 && m_board[(rank - i) * 8 + (file + i)].checkOccupied() == false)
                addMove(m_board[(rank - i) * 8 + (file + i)]);
            else if (file + i < 8 && rank - i > -1 && 
                     m_board[(rank - i) * 8 + (file + i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank - i) * 8 + (file + i)]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // down left
        {
            if (file - i > -1 && rank - i > -1 && m_board[(rank - i) * 8 + (file - i)].checkOccupied() == false)
                addMove(m_board[(rank - i) * 8 + (file - i)]);
            else if (file - i > -1 && rank - i > -1 && 
                     m_board[(rank - i) * 8 + (file - i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank - i) * 8 + (file - i)]);
                break;
            }
            else break;
        }
        break;
    }
    case QUEEN:
    {
        // Combinaison des logiques ROOK et BISHOP
        for (int i = rank + 1; i < 8; i++)  // up
        {
            if (m_board[i * 8 + file].checkOccupied() == false)
                addMove(m_board[i * 8 + file]);
            else if (m_board[i * 8 + file].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[i * 8 + file]);
                break;
            }
            else break;
        }
        for (int i = rank - 1; i > -1; i--) // down
        {
            if (m_board[i * 8 + file].checkOccupied() == false)
                addMove(m_board[i * 8 + file]);
            else if (m_board[i * 8 + file].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[i * 8 + file]);
                break;
            }
            else break;
        }
        for (int i = file + 1; i < 8; i++) // right
        {
            if (m_board[rank * 8 + i].checkOccupied() == false)
                addMove(m_board[rank * 8 + i]);
            else if (m_board[rank * 8 + i].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[rank * 8 + i]);
                break;
            }
            else break;
        }
        for (int i = file - 1; i > -1; i--) // left
        {
            if (m_board[rank * 8 + i].checkOccupied() == false)
                addMove(m_board[rank * 8 + i]);
            else if (m_board[rank * 8 + i].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[rank * 8 + i]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // up right
        {
            if (file + i < 8 && rank + i < 8 && m_board[(rank + i) * 8 + (file + i)].checkOccupied() == false)
                addMove(m_board[(rank + i) * 8 + (file + i)]);
            else if (file + i < 8 && rank + i < 8 && 
                     m_board[(rank + i) * 8 + (file + i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank + i) * 8 + (file + i)]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // up left
        {
            if (file - i > -1 && rank + i < 8 && m_board[(rank + i) * 8 + (file - i)].checkOccupied() == false)
                addMove(m_board[(rank + i) * 8 + (file - i)]);
            else if (file - i > -1 && rank + i < 8 && 
                     m_board[(rank + i) * 8 + (file - i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank + i) * 8 + (file - i)]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // down right
        {
            if (file + i < 8 && rank - i > -1 && m_board[(rank - i) * 8 + (file + i)].checkOccupied() == false)
                addMove(m_board[(rank - i) * 8 + (file + i)]);
            else if (file + i < 8 && rank - i > -1 && 
                     m_board[(rank - i) * 8 + (file + i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank - i) * 8 + (file + i)]);
                break;
            }
            else break;
        }
        for (int i = 1; i < 8; i++) // down left
        {
            if (file - i > -1 && rank - i > -1 && m_board[(rank - i) * 8 + (file - i)].checkOccupied() == false)
                addMove(m_board[(rank - i) * 8 + (file - i)]);
            else if (file - i > -1 && rank - i > -1 && 
                     m_board[(rank - i) * 8 + (file - i)].getPiece().getColor() == oppositeColor)
            {
                addMove(m_board[(rank - i) * 8 + (file - i)]);
                break;
            }
            else break;
        }
        break;
    }
    case KING:
    {
        if (file + 1 < 8 && rank + 1 < 8 && m_board[(rank + 1) * 8 + (file + 1)].getPiece().getColor() != color)
            addMove(m_board[(rank + 1) * 8 + (file + 1)]);
        if (file + 1 < 8 && rank - 1 > -1 && m_board[(rank - 1) * 8 + (file + 1)].getPiece().getColor() != color)
            addMove(m_board[(rank - 1) * 8 + (file + 1)]);
        if (file - 1 > -1 && rank + 1 < 8 && m_board[(rank + 1) * 8 + (file - 1)].getPiece().getColor() != color)
            addMove(m_board[(rank + 1) * 8 + (file - 1)]);
        if (file - 1 > -1 && rank - 1 > -1 && m_board[(rank - 1) * 8 + (file - 1)].getPiece().getColor() != color)
            addMove(m_board[(rank - 1) * 8 + (file - 1)]);
        if (file + 1 < 8 && m_board[rank * 8 + (file + 1)].getPiece().getColor() != color)
            addMove(m_board[rank * 8 + (file + 1)]);
        if (file - 1 > -1 && m_board[rank * 8 + (file - 1)].getPiece().getColor() != color)
            addMove(m_board[rank * 8 + (file - 1)]);
        if (rank + 1 < 8 && m_board[(rank + 1) * 8 + file].getPiece().getColor() != color)
            addMove(m_board[(rank + 1) * 8 + file]);
        if (rank - 1 > -1 && m_board[(rank - 1) * 8 + file].getPiece().getColor() != color)
            addMove(m_board[(rank - 1) * 8 + file]);

        // short castle
        if (color == WHITE && m_short_castle_white == true && 
            m_board[0 * 8 + 5].checkOccupied() == false && 
            m_board[0 * 8 + 6].checkOccupied() == false)
            addMove(m_board[0 * 8 + 6]);
        if (color == BLACK && m_short_castle_black == true && 
            m_board[7 * 8 + 5].checkOccupied() == false && 
            m_board[7 * 8 + 6].checkOccupied() == false)
            addMove(m_board[7 * 8 + 6]);
        // long castle
        if (color == WHITE && m_long_castle_white == true && 
            m_board[0 * 8 + 1].checkOccupied() == false &&
            m_board[0 * 8 + 2].checkOccupied() == false && 
            m_board[0 * 8 + 3].checkOccupied() == false)
            addMove(m_board[0 * 8 + 2]);
        if (color == BLACK && m_long_castle_black == true && 
            m_board[7 * 8 + 1].checkOccupied() == false &&
            m_board[7 * 8 + 2].checkOccupied() == false && 
            m_board[7 * 8 + 3].checkOccupied() == false)
            addMove(m_board[7 * 8 + 2]);
        break;
    }
    }
}

void Chessboard::computeInitialZobrist() {
    m_current_zobrist_hash = 0;

    // 1. Placement des pièces
    for (int i = 0; i < 64; i++) {
        const Piece& p = m_board[i].getPiece();
        if (p.getType() != NONE) {
            int piece_idx = p.getZobristIndex();
            m_current_zobrist_hash ^= Zobrist::PIECE_KEYS[i][piece_idx];
        }
    }

    // 2. Trait
    if (m_turn == BLACK) {
        m_current_zobrist_hash ^= Zobrist::BLACK_TO_MOVE;
    }

    // 3. Droits de roque (Encodage sur 4 bits de 0 à 15)
    int castling_idx = 0;
    if (m_short_castle_white) castling_idx |= 1; // Bit 0
    if (m_long_castle_white)  castling_idx |= 2; // Bit 1
    if (m_short_castle_black) castling_idx |= 4; // Bit 2
    if (m_long_castle_black)  castling_idx |= 8; // Bit 3

    m_current_zobrist_hash ^= Zobrist::CASTLING_KEYS[castling_idx];

    // 4. Case en passant
    if (m_en_passant && m_en_passant_file >= 0 && m_en_passant_file < 8) {
        m_current_zobrist_hash ^= Zobrist::EN_PASSANT_KEYS[m_en_passant_file];
    }
}

uint64_t Chessboard::computeZobristFromScratch() const {
    uint64_t hash = 0;

    // 1. Placement des pièces
    for (int i = 0; i < 64; i++) {
        const Piece& p = m_board[i].getPiece();
        if (p.getType() != NONE) {
            int piece_idx = p.getZobristIndex();
            hash ^= Zobrist::PIECE_KEYS[i][piece_idx];
        }
    }

    // 2. Trait
    if (m_turn == BLACK) {
        hash ^= Zobrist::BLACK_TO_MOVE;
    }

    // 3. Droits de roque
    int castling_idx = 0;
    if (m_short_castle_white) castling_idx |= 1;
    if (m_long_castle_white)  castling_idx |= 2;
    if (m_short_castle_black) castling_idx |= 4;
    if (m_long_castle_black)  castling_idx |= 8;
    hash ^= Zobrist::CASTLING_KEYS[castling_idx];

    // 4. Case en passant
    if (m_en_passant && m_en_passant_file >= 0 && m_en_passant_file < 8) {
        hash ^= Zobrist::EN_PASSANT_KEYS[m_en_passant_file];
    }

    return hash;
}