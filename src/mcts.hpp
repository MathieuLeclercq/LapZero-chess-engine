#pragma once

#include "board_rollback.hpp"
#include "chessboard.hpp"
#include <vector>
#include <memory>
#include <utility>
#include <cstdint>
#include <mutex>
#include <random>
#include <atomic>
#include <string>

#include "evaluation_cache.hpp"
#include "evaluator.hpp"
#include "mcts_reservation.hpp"
#include "mcts_wave.hpp"
#include "search_timing.hpp"

class SearchExecutor;
class MCTSTestAccess;

struct MCTSNode {
    int visit_count;
    int move_idx;
    float prior;
    float total_value;

    // Valeur statique du reseau a l'expansion, du point de vue du noeud (meme
    // convention que total_value). Elle ne bouge plus ensuite : c'est le V
    // publie par l'UCI, a ne pas confondre avec q_value(), la moyenne cherchee.
    float network_value = 0.0f;

    // Virtual loss, approche LC0 : nombre de descentes en cours passant par ce
    // noeud. N'entre QUE dans le denominateur du terme U de ucb_score, jamais
    // dans q_value(), sans quoi Q se diluerait vers zero et avantagerait les
    // noeuds perdants.
    // L'etat publie la liste d'enfants. Un lecteur ne consulte children
    // qu'apres un load-acquire ayant observe Expanded.
    std::atomic<NodeState> state{NodeState::Unexpanded};
    std::atomic<std::uint32_t> n_in_flight{0};

    MCTSNode* parent;
    std::vector<std::pair<int, std::unique_ptr<MCTSNode>>> children;

    MCTSNode(float prior, int move_idx = -1, MCTSNode* parent = nullptr);
    float q_value() const;
    float ucb_score(float exploration_factor, float parent_q, float fpu_reduction) const;
    MCTSNode* find_child(int idx) const;
    bool has_child(int idx) const;
    std::unique_ptr<MCTSNode> extract_child(int idx);
};


struct MoveStats {
    int move_idx;
    int visits;
    float q_value;
    float prior;
    float v_value;
};


// Instrumentation de la recherche. nn_calls compte les positions evaluees,
// nn_batches les appels physiques a l'evaluateur. Leur ratio donne le
// remplissage moyen des lots.
struct SearchCounters {
    uint64_t nn_calls = 0;
    uint64_t nn_batches = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_misses = 0;
    uint64_t tt_position_matches = 0;
    uint64_t tt_rule50_rejects = 0;
    uint64_t tt_context_rejects = 0;
    uint64_t tt_history_rejects = 0;
    uint64_t terminal_hits = 0;
    uint64_t waves = 0;
    uint64_t leaf_collisions = 0;
    uint64_t completed_simulations = 0;
};


// Rapport de parcours de l'arbre, sur le modele de PerftReport.
struct TreeReport {
    uint64_t nodes = 0;
    uint64_t max_depth = 0;
    uint64_t violations = 0;
    uint64_t en_vol = 0;   // noeuds dont n_in_flight != 0 apres la recherche
    uint64_t pending = 0;
    uint64_t root_visits = 0;
    std::vector<std::string> messages;
};


// Reglages de divergence de la collecte. Les valeurs par defaut sont celles
// historiques ; toute modification change la recherche et exige une validation
// qualite par le banc de puzzles.
struct SearchTuning {
    int virtual_loss = 1;             // unites de n_in_flight par descente
    float fpu_reduction = 0.30f;      // coefficient du terme FPU
    int collision_attempt_factor = 4; // tentatives = facteur * slots + workers
};

class MCTS {
    friend class MCTSTestAccess;

private:
    static constexpr size_t DEFAULT_TT_SIZE = 2097143;
    EvaluationCache m_cache;
    int m_cache_history_depth;
    std::vector<float> m_eval_tensor;
    std::vector<float> m_eval_policy;
    std::unique_ptr<MCTSNode> m_analysis_root;
    mutable std::mutex m_mutex;
    Evaluator* m_evaluator;
    std::mt19937 m_noise_rng;
    bool m_timing_enabled = false;
    SearchTiming m_last_timing;
    bool m_fixed_batch = false;
    SearchTuning m_tuning;
    std::unique_ptr<SearchExecutor> m_search_executor;
    std::vector<WorkerContext> m_worker_contexts;

    // Injection de panne des vagues, reservee aux tests. Nulle en production
    // et non exposee dans les bindings.
    const WaveTestHooks* m_wave_test_hooks = nullptr;

    // Atomiques parce que le self-play appelle advance_to_leaf depuis une region
    // OpenMP a 8 fils sur une instance de MCTS partagee
    // (selfplay_manager.cpp:376-387). L'ordre relache suffit : on ne lit ces
    // compteurs qu'apres la recherche, et un incremente relache coute quelques
    // dizaines de cycles contre 2,7 ms d'inference.
    std::atomic<uint64_t> m_nn_calls{ 0 };
    std::atomic<uint64_t> m_nn_batches{ 0 };
    std::atomic<uint64_t> m_tt_hits{ 0 };
    std::atomic<uint64_t> m_tt_misses{ 0 };
    std::atomic<uint64_t> m_tt_position_matches{ 0 };
    std::atomic<uint64_t> m_tt_rule50_rejects{ 0 };
    std::atomic<uint64_t> m_tt_context_rejects{ 0 };
    std::atomic<uint64_t> m_tt_history_rejects{ 0 };
    std::atomic<uint64_t> m_terminal_hits{ 0 };
    std::atomic<uint64_t> m_waves{ 0 };
    std::atomic<uint64_t> m_leaf_collisions{ 0 };
    std::atomic<uint64_t> m_completed_simulations{ 0 };

public:
    MCTS(Evaluator* evaluator, size_t tt_size = DEFAULT_TT_SIZE,
         int cache_history_depth = DEFAULT_CACHE_HISTORY_DEPTH);
    ~MCTS();

    void step_analysis(Chessboard& board, int num_simulations, float c_puct,
                       int batch_size = 0, int worker_count = 1);
    void reset_analysis();
    void update_root(int move_idx);
    float get_root_q() const;
    std::vector<MoveStats> get_analysis_results() const;
    // Remise a froid de la TT, au repos seulement. Le pool de workers reste en
    // place : les threads ne sont jamais recrees par un reset.
    void clear_evaluation_cache();
    std::vector<float> mcts_search(Chessboard& board, int num_simulations, float c_puct,
                                   bool add_dirichlet, int batch_size = 0,
                                   int worker_count = 1);
    float expand_node_single(MCTSNode* node, Chessboard& board,
                             SearchTiming* timing = nullptr);
    bool apply_move_by_index(Chessboard& board, int idx);
    void add_dirichlet_noise(MCTSNode* root, float epsilon);

    // Observabilite, definie dans mcts_observe.cpp.
    SearchCounters get_counters() const;
    void reset_counters();
    TreeReport inspect_tree() const;
    TreeReport inspect_tree(const MCTSNode* root) const;
    void set_timing_enabled(bool enabled);
    SearchTiming get_last_timing() const;
    // Forme de lot fixe : les vagues sont paddees a batch_size pour eviter la
    // re-planification d'ONNX Runtime a chaque changement de forme. Les
    // sorties des positions dupliquees sont ignorees.
    void set_fixed_batch(bool enabled);
    bool fixed_batch() const;
    void set_tuning(const SearchTuning& tuning);
    SearchTuning get_tuning() const;

    // recherche mcts asynchrone
    MCTSNode* advance_to_leaf(MCTSNode* root, Chessboard& board, float c_puct,
                              int& moves_played,
                              PathReservation& reservation,
                              SearchWorkCounters* work = nullptr);
    void expand_and_backup(MCTSNode* leaf_node, Chessboard& board,
                           const float* policy, float value,
                           PathReservation& reservation);

private:
    void backup(MCTSNode* node, float value, SearchTiming* timing = nullptr);
    EvaluationCacheKey make_cache_key(const Chessboard& board) const;
    TTProbe probe_tt(const EvaluationCacheKey& key,
                     SearchTiming* timing = nullptr) const;
    void record_tt_probe(TTProbeStatus status);
    void store_tt(const EvaluationCacheKey& key,
                  const std::vector<int>& legal_indices,
                  const float* policy, float value,
                  SearchTiming* timing = nullptr);
    // Chaque coup joue est arme sur le garde : la descente peut ainsi etre
    // restauree meme si elle est interrompue par une erreur.
    std::pair<MCTSNode*, int> select_leaf(MCTSNode* root, Chessboard& board,
                                         float c_puct,
                                         BoardRollback& rollback,
                                         SearchTiming* timing = nullptr,
                                         SearchWorkCounters* work = nullptr);
    void expand_and_backup_prepared(MCTSNode* leaf_node,
                                    const std::vector<int>& legal_indices,
                                    const EvaluationCacheKey& key,
                                    const float* policy, float value,
                                    PathReservation& reservation,
                                    SearchTiming* timing = nullptr);
    LeafWork collect_wave_leaf(
        MCTSNode* root, WorkerContext& context, float c_puct,
        const std::atomic<bool>& cancelled,
        const WaveTestHooks* hooks = nullptr);
    std::vector<LeafWork> collect_wave(
        MCTSNode* root, float c_puct, std::size_t slots,
        std::size_t worker_count, SearchExecutor& executor,
        std::vector<WorkerContext>& contexts,
        const std::atomic<bool>& cancelled,
        const WaveTestHooks* hooks = nullptr);

    // Noyau unique de recherche, defini dans mcts_batch.cpp.
    // batch_size == 0 : boucle sequentielle historique, conservee telle quelle.
    // batch_size >= 1 : boucle batchee avec virtual loss.
    void run_search(MCTSNode* root, Chessboard& board, int simulations,
                    float c_puct, int batch_size,
                    SearchTiming* timing = nullptr);
    void run_search_waves(MCTSNode* root, const Chessboard& board,
                          int simulations, float c_puct, int batch_size,
                          int worker_count, SearchTiming* timing = nullptr);

};
