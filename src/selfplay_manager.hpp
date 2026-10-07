#pragma once
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <cstdint>
#include "chessboard.hpp"
#include "evaluator.hpp"
#include "mcts.hpp"
#include "selfplay_timing.hpp"


struct GameResult {
    std::vector<float> flat_states; // Taille : [NbCoups * 119 * 64]
    std::vector<float> flat_policies; // Taille : [NbCoups * 4672]
    float final_outcome;
    int move_count; // Pour savoir comment découper le vecteur plat
    int total_real_moves;
    int end_reason;
};

// Compteurs de diagnostic d'une generation self-play. Une seule generation par
// appel : les compteurs repartent de zero au debut de generate_games.
struct SelfPlayStats {
    std::uint64_t games_started = 0;    // parties reellement demarrees
    std::uint64_t games_completed = 0;  // parties terminees et enregistrees
    std::uint32_t active_slots = 0;     // places avec une partie en cours
    std::uint64_t new_plies = 0;        // coups joues par la recherche
    std::uint64_t replayed_plies = 0;   // historique de puzzle rejoue
};

// Conclusion d'une partie terminee : issue du point de vue du camp au trait et
// raison de fin (0 mat, 1 pat, 2 repetition, 3 cinquante coups, 4 materiel
// insuffisant, 5 longueur maximale). Le mat prime quand les conditions se
// recouvrent, comme dans la recherche.
struct GameConclusion {
    float final_outcome = 0.0f;
    int end_reason = 1;
};

inline GameConclusion conclure_partie(Chessboard& board,
                                      bool longueur_max_atteinte) {
    if (!board.hasAnyLegalMove() && board.isInCheck()) {
        return {board.getTurn() == WHITE ? -1.0f : 1.0f, 0};
    }
    if (longueur_max_atteinte) return {0.0f, 5};
    if (board.checkThreefoldRepetition()) return {0.0f, 2};
    if (board.getHalfMoveClock() >= 100) return {0.0f, 3};
    if (board.checkInsufficientMaterial()) return {0.0f, 4};
    return {0.0f, 1};
}

struct ThreadLocalBuffer {
    std::vector<MCTSNode*> leaves;
    std::vector<int> game_indices;
    std::vector<int> moves_played;
    std::vector<PathReservation> reservations;
    std::vector<float> tensors;
    std::vector<float> tensor_scratch;

    void clear() {
        leaves.clear();
        game_indices.clear();
        moves_played.clear();
        reservations.clear();
    }
};

class SelfPlayManager {
private:

    friend class SelfPlayTestAccess;

    // constantes de config
    static constexpr float NORMAL_EPSILON = 0.12f;
    static constexpr float TACTICAL_EPSILON = 0.30f;
    static constexpr int TACTICAL_FIRST_MOVE_SIMS = 4000;
    // Longueur maximale d'une partie avant la nulle forcee, en demi-coups
    // joues en self-play : l'historique rejoue d'un puzzle n'y compte pas.
    static constexpr int MAX_SELFPLAY_PLIES = 400;

    int m_num_concurrent_games;
    int m_slow_sims;
    int m_fast_sims;
    float m_slow_ratio;
    std::vector<int> m_sims_target;  // sims à faire pour le coup en cours
    std::vector<char> m_is_slow_move;

    // Bruit de Dirichlet du a la racine du coup en cours, une seule fois par
    // coup. Il est applique des que les enfants de la racine existent : les
    // positions servies par la table n'en ont pas a la preparation.
    std::vector<float> m_pending_epsilon;

    // Places actives : une place sans partie ne participe ni aux descentes, ni
    // a la condition de lot plein, ni a la taille utile du batch.
    std::vector<char> m_slot_active;
    SelfPlayStats m_stats;

    // Levier de test : fin de partie imposee apres un nombre de plies connu,
    // 0 pour desactiver. Toujours a zero en production.
    std::vector<int> m_forced_end_plies;

    // Bruit de Dirichlet des coups rapides. Vrai : mode historique, les deux
    // types de coups recoivent NORMAL_EPSILON. Faux : seule la classification
    // slow/fast decide, les coups rapides n'ont aucun bruit. Le boost du
    // premier coup de puzzle reste prioritaire dans les deux modes.
    bool m_fast_dirichlet_noise = true;

    // Diagnostic de debit, desactive par defaut. m_last_batch_rows sert au
    // comptage des changements de taille entre deux appels physiques.
    int m_diagnostics_mode = 0;
    SelfPlayTiming m_timing;
    std::uint64_t m_last_batch_rows = 0;

    Evaluator* m_evaluator;

    // L'état complet des parties en cours
    std::vector<Chessboard> m_boards;
    std::vector<std::unique_ptr<MCTSNode>> m_roots;
    std::unique_ptr<MCTS> m_shared_mcts;

    // Suivi de l'avancement de chaque arbre (combien de simulations terminées pour ce coup)
    std::vector<int> m_sims_completed;

    // Données accumulées pour l'entraînement final
    std::vector<GameResult> m_finished_games;
    std::vector<std::vector<std::vector<float>>> m_game_states;
    std::vector<std::vector<std::vector<float>>> m_game_policies;

    // Buffers pour le GPU
    std::vector<float> m_batch_input;
    std::vector<float> m_batch_policies;
    std::vector<float> m_batch_values;

    // Suivi de l'état de la boucle asynchrone
    std::vector<MCTSNode*> m_waiting_leaves;
    std::vector<int> m_waiting_game_indices;
    std::vector<int> m_waiting_moves_played;
    std::vector<PathReservation> m_waiting_reservations;
    std::vector<char> m_is_waiting;

    std::mt19937 m_rng{ std::random_device{}() };

    // pour entrainement tactique sur puzzles lichess
    // Renfort tactique du PREMIER coup d'une partie amorcee par un puzzle :
    // TACTICAL_FIRST_MOVE_SIMS simulations et TACTICAL_EPSILON de bruit. Le
    // drapeau est consomme dans play_best_move une fois ce coup joue, apres
    // quoi la partie redevient une partie normale, budget de recherche comme
    // bruit de Dirichlet.
    std::vector<char> m_tactical_boost;

    // Demi-coups rejoues au depart d'une partie de puzzle, 0 sinon. Exclus du
    // plafond de longueur, mais comptes dans total_real_moves : la longueur
    // enregistree reste celle de la partie d'echecs complete.
    std::vector<int> m_replayed_plies;

    // Longueur de partie comptee pour le plafond, et test du plafond.
    int selfplay_plies(int game_idx) const;
    bool longueur_max_atteinte(int game_idx) const;

    // Epsilon du bruit de Dirichlet du coup prepare. Le renfort du premier
    // coup de puzzle prime ; sinon un coup rapide ne recoit du bruit que si
    // le mode historique est actif. La classification m_is_slow_move decide,
    // jamais le nombre de simulations.
    float epsilon_for_move(int game_idx) const;

    struct TacticalPuzzle {
        std::string start_fen;
        std::vector<std::string> moves; // UCI, jusqu'à la position du puzzle incluse
    };
    std::vector<TacticalPuzzle> m_tactical_puzzles;

    // Positions de depart imposees, une FEN par ligne, pour les bancs de
    // diagnostic. Non vide, elle remplace la position initiale et les
    // puzzles : chaque partie part d'une position tiree au hasard, sans
    // renfort tactique. Vide en production.
    std::vector<std::string> m_start_fens;


public:
    SelfPlayManager(Evaluator* evaluator, int num_concurrent_games,
                    int slow_sims, int fast_sims, float slow_ratio,
                    size_t tt_size = 2097143,
                    const std::string& puzzles_path = "../training_data/puzzles_train.txt",
                    const std::string& start_positions_path = "");
    std::vector<GameResult> generate_games(int total_games_to_play);
    SelfPlayStats get_stats() const;

    // 0 desactive, 1 phases de premier niveau, 2 phases plus detail par worker.
    void set_diagnostics_mode(int mode);
    int get_diagnostics_mode() const;
    SelfPlayTiming get_timing() const;

    // A appeler avant generate_games. Ne change pas la distribution du bruit,
    // seulement son application aux coups rapides.
    void set_fast_dirichlet_noise(bool enabled);

private:
    void reset_game(int game_idx);
    void demarrer_slot(int game_idx);
    void play_best_move(int game_idx);
    void roll_next_move(int game_idx);
    void execute_gpu_batch();
    void apply_pending_noise(int game_idx);
    void load_tactical_puzzles(const std::string& filepath);
    void load_start_positions(const std::string& filepath);
    // Enveloppe les expansions de racine du gestionnaire pour y accrocher les
    // compteurs et les sous-durees ONNX quand le diagnostic est actif.
    void expand_root(int game_idx);
};
