#pragma once
#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include "piece.hpp"
#include "square.hpp"
#include "move.hpp"

enum GameState
{
    ONGOING,
    CHECKMATE,
    STALEMATE,
    DRAW_REPETITION,
    DRAW_50_MOVES,
    DRAW_INSUFF_MATERIAL
};

struct StateSnapshot {
    bool short_castle_white;
    bool long_castle_white;
    bool short_castle_black;
    bool long_castle_black;
    bool en_passant;
    int en_passant_file;
    int half_move_clock;
    int white_king_file;
    int white_king_rank;
    int black_king_file;
    int black_king_rank;
    uint64_t zobrist_hash;
    GameState current_state;
};

struct EvaluationCacheKey {
    uint64_t position_hash = 0;
    uint64_t current_context_hash = 0;
    uint64_t history_hash = 0;
    uint64_t combined_hash = 0;
    uint16_t half_move_clock = 0;
    uint8_t repetition_category = 0;
    uint8_t total_moves_bucket = 0;
};

// Encode un coup en index de policy (0 a 4671) du point de vue du camp donne.
// Renvoie -1 si le coup n'est pas encodable. Seul encodeur du moteur : le
// binding Python et Chessboard::encodeMove passent par lui.
int encodeMoveIndex(int orig_f, int orig_r, int dest_f, int dest_r,
                    PieceType promotion, bool is_black);

class Chessboard
{
    private:
    std::array<Square, 64> m_board;

    int m_white_king_file = 4; // utile pour checkForCheck() : pas besoin de chercher le roi à chaque fois
    int m_white_king_rank = 0;
    int m_black_king_file = 4;
    int m_black_king_rank = 7;
    int m_en_passant_file = -1;
    int m_half_move_clock = 0;

    bool m_short_castle_white = true;
    bool m_long_castle_white = true;
    bool m_short_castle_black = true;
    bool m_long_castle_black = true;
    bool m_en_passant = false;
    bool m_amnesia_mode = false;

    GameState m_current_state = ONGOING;
    Color m_turn = WHITE;

    std::vector<Move> m_moveHistory;
    std::vector<std::array<Square, 64>> m_boardHistory;
    int m_initial_ply_offset = 0;  // si on charge une partie en cours de route
    std::vector<StateSnapshot> m_snapshotHistory;
    uint64_t m_current_zobrist_hash = 0;


    // Vérifie uniquement que le roi ne traverse pas de case contrôlée.
    // Les droits de roque, les cases vides et la présence de la tour
    // sont vérifiés en amont par getNaiveLegalMoves().
    bool isCastleSafe(int orig_file, int orig_rank, int file, int rank);
    bool isMoveSafe(int orig_f, int orig_r, int dest_f, int dest_r, bool is_en_passant, bool is_king_move);
    void evaluateGameState();
    void computeInitialZobrist();

    public:
        // constructors
        Chessboard();
        
        // getters
        uint64_t getZobristHash() const { return m_current_zobrist_hash; }
        uint64_t computeZobristFromScratch() const;
        EvaluationCacheKey getEvaluationCacheKey(int history_depth) const;
        int getNumberOfOccupiedSquares() const;
        int getHalfMoveClock() const;
        int encodeMove(const Move& move) const;
        // Inverse de encodeMove : decode un index de policy (0 a 4671) en coup
        // absolu, orientation du camp au trait et promotion dame incluses.
        // Le seul decodeur d'index du moteur. L'index doit appartenir a
        // [0, 4671] ; hors bornes, le decodage n'est pas defini, comme du
        // temps de la table Python.
        Move decodeMoveIndex(int index) const;
       
        Color getTurn() const;
        GameState getGameState() const;
        Square& getSquare(int file, int rank);
        const Square& getSquare(int file, int rank) const;

        std::vector<int> getLegalMoveIndices();
        void getAlphaZeroTensor(std::vector<float>& tensor) const;
        std::vector<Move> getAllLegalMoves();
        void getLegalMovesForSquare(int file, int rank, std::vector<Move>& result,
            std::vector<Move>& pseudo_buffer,
            int filter_dest_file = -1, int filter_dest_rank = -1);
        void getNaiveLegalMoves(int file, int rank, std::vector<Move>& pseudo_buffer) const;

        std::vector<Move>& getMoveHistory();
        const std::vector<Move>& getMoveHistory() const;
        std::vector<std::array<Square, 64>>& getBoardHistory();
        const std::vector<std::array<Square, 64>>& getBoardHistory() const;
        int getInitialPlyOffset() const;

        void checkEnPassant();
        void print() const;
        void print(std::array<Square, 64> some_board) const;
        void printPly() const;
        void loadFEN(const std::string& fen);
        std::string toFEN() const;
        bool checkThreefoldRepetition() const;
        bool isInCheck() const;
        bool hasAnyLegalMove();
        bool checkInsufficientMaterial() const;
        void setAmnesiaMode(bool amnesia);

        // setters
        void clear();
        void setStartupPieces();
        void setKiwipete();
        void setBoard(std::array<Square, 64> some_board);
        bool movePiece(int orig_file, int orig_rank, int file, int rank, 
                       PieceType promotion = NONE, bool check_game_end = true);
        bool movePiece(std::string orig_square, std::string square);
        bool movePieceSAN(std::string san);
        // Applique un coup en notation UCI ("e2e4", "a7a8q").
        // Renvoie false si la chaîne est mal formée ou le coup illégal.
        bool movePieceUCI(const std::string& uci);
        void undoMove();
        void updateHistory(const Move& move);
        void updateCastleFlags();
        void applyPromotion(Square& second_square, PieceType force_promotion = NONE);
        void updateStateSnapshot();
};
