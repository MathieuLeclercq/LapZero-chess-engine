#pragma once

#include "chessboard.hpp"
#include "evaluation_cache.hpp"
#include "mcts_reservation.hpp"
#include "search_timing.hpp"

#include <functional>
#include <vector>

struct MCTSNode;

enum class LeafKind {
    Network,
    Terminal,
    Collision,
};

struct LeafWork {
    LeafKind kind = LeafKind::Collision;
    MCTSNode* node = nullptr;
    PathReservation reservation;
    EvaluationCacheKey key;
    std::vector<int> legal_moves;
    std::vector<float> tensor;
    float terminal_value = 0.0f;
    bool known_terminal = false;
};

struct WorkerContext {
    Chessboard board;
    std::vector<LeafWork> results;
    SearchTiming timing;
};

// Points de synchronisation reserves aux tests d'interleaving. Ils restent
// nuls en production et ne sont pas exposes dans les bindings Python.
struct WaveTestHooks {
    std::function<void(MCTSNode*)> before_claim;
    std::function<void(MCTSNode*)> before_tt_publish;
    // Injection de panne de la fusion : appelee apres la collecte avant tout
    // transfert, puis apres chaque resultat transfere vers le vecteur fusionne.
    // La seconde permet de tester un transfert partiel.
    std::function<void()> before_merge;
    std::function<void(std::size_t)> after_transfer;
};
