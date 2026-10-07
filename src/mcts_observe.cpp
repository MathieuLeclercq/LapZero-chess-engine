// Observabilite de la recherche : compteurs et parcours d'arbre.
//
// Separe de mcts.cpp, qui fait deja 512 lignes, et sur le modele de perft.cpp :
// un parcours de diagnostic a sa propre responsabilite et son propre fichier.

#include "mcts.hpp"

#include <algorithm>
#include <cmath>

void MCTS::set_timing_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_timing_enabled = enabled;
    if (!enabled) {
        m_last_timing = SearchTiming{};
    }
}

SearchTiming MCTS::get_last_timing() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_last_timing;
}

SearchCounters MCTS::get_counters() const {
    SearchCounters c;
    c.nn_calls = m_nn_calls.load(std::memory_order_relaxed);
    c.nn_batches = m_nn_batches.load(std::memory_order_relaxed);
    c.tt_hits = m_tt_hits.load(std::memory_order_relaxed);
    c.tt_misses = m_tt_misses.load(std::memory_order_relaxed);
    c.tt_position_matches = m_tt_position_matches.load(std::memory_order_relaxed);
    c.tt_rule50_rejects = m_tt_rule50_rejects.load(std::memory_order_relaxed);
    c.tt_context_rejects = m_tt_context_rejects.load(std::memory_order_relaxed);
    c.tt_history_rejects = m_tt_history_rejects.load(std::memory_order_relaxed);
    c.terminal_hits = m_terminal_hits.load(std::memory_order_relaxed);
    c.waves = m_waves.load(std::memory_order_relaxed);
    c.leaf_collisions = m_leaf_collisions.load(std::memory_order_relaxed);
    c.completed_simulations =
        m_completed_simulations.load(std::memory_order_relaxed);
    return c;
}

void MCTS::reset_counters() {
    m_nn_calls.store(0, std::memory_order_relaxed);
    m_nn_batches.store(0, std::memory_order_relaxed);
    m_tt_hits.store(0, std::memory_order_relaxed);
    m_tt_misses.store(0, std::memory_order_relaxed);
    m_tt_position_matches.store(0, std::memory_order_relaxed);
    m_tt_rule50_rejects.store(0, std::memory_order_relaxed);
    m_tt_context_rejects.store(0, std::memory_order_relaxed);
    m_tt_history_rejects.store(0, std::memory_order_relaxed);
    m_terminal_hits.store(0, std::memory_order_relaxed);
    m_waves.store(0, std::memory_order_relaxed);
    m_leaf_collisions.store(0, std::memory_order_relaxed);
    m_completed_simulations.store(0, std::memory_order_relaxed);
}


// Nombre maximum de messages conserves, sur le modele de MAX_PERFT_MESSAGES :
// eviter de noyer la sortie quand un bug se declenche sur des milliers de noeuds.
static constexpr size_t MAX_TREE_MESSAGES = 20;

static void ajouter_violation(TreeReport& rapport, const std::string& message) {
    rapport.violations++;
    if (rapport.messages.size() < MAX_TREE_MESSAGES) {
        rapport.messages.push_back(message);
    }
}

static void visiter(const MCTSNode* node, uint64_t profondeur, TreeReport& rapport) {
    rapport.nodes++;
    const std::uint32_t in_flight =
        node->n_in_flight.load(std::memory_order_relaxed);
    const NodeState state = node->state.load(std::memory_order_acquire);
    if (in_flight != 0) {
        // Une reservation non liberee est une fuite, pas un etat normal : au
        // repos, aucune descente ne doit etre en vol. Le compteur en_vol reste
        // le nombre de noeuds concernes, pas la somme des unites.
        rapport.en_vol++;
        ajouter_violation(rapport,
            "noeud en vol au repos, n_in_flight " + std::to_string(in_flight));
    }
    if (state == NodeState::Pending) {
        rapport.pending++;
        ajouter_violation(rapport, "noeud Pending au repos");
    }
    rapport.max_depth = std::max(rapport.max_depth, profondeur);

    if (!std::isfinite(node->prior) ||
        !std::isfinite(node->total_value) ||
        !std::isfinite(node->q_value())) {
        ajouter_violation(rapport, "statistique non finie");
    }
    if (node->visit_count < 0) {
        ajouter_violation(rapport, "nombre de visites negatif");
    }

    const bool has_children = !node->children.empty();
    if (state == NodeState::Unexpanded && has_children) {
        ajouter_violation(rapport, "noeud Unexpanded avec des enfants");
    }
    if (state == NodeState::Pending && has_children) {
        ajouter_violation(rapport, "noeud Pending avec des enfants");
    }
    if (state == NodeState::Expanded && !has_children) {
        ajouter_violation(rapport, "noeud Expanded sans enfant");
    }
    if (state == NodeState::Terminal && has_children) {
        ajouter_violation(rapport, "noeud Terminal avec des enfants");
    }

    if (!has_children) return;

    std::vector<int> vus;
    vus.reserve(node->children.size());
    uint64_t somme_visites = 0;
    float somme_priors = 0.0f;

    for (const auto& pair : node->children) {
        const int idx = pair.first;
        const MCTSNode* enfant = pair.second.get();

        // Aucun enfant duplique. C'est le controle qui attrape le piege du
        // batching : une feuille collectee qui recevrait un second jeu
        // d'enfants pendant la meme collecte.
        if (std::find(vus.begin(), vus.end(), idx) != vus.end()) {
            ajouter_violation(rapport,
                "enfant duplique, move_idx " + std::to_string(idx));
        }
        vus.push_back(idx);

        if (idx < 0 || idx > 4671) {
            ajouter_violation(rapport,
                "move_idx hors bornes : " + std::to_string(idx));
        }

        if (enfant->parent != node) {
            ajouter_violation(rapport,
                "pointeur parent incoherent, move_idx " + std::to_string(idx));
        }

        if (!std::isfinite(enfant->prior) ||
            !std::isfinite(enfant->total_value) ||
            !std::isfinite(enfant->q_value())) {
            ajouter_violation(rapport,
                "statistique enfant non finie, move_idx " +
                std::to_string(idx));
        }

        if (enfant->visit_count >= 0) {
            somme_visites += static_cast<uint64_t>(enfant->visit_count);
        }
        somme_priors += enfant->prior;
    }

    // Conservation des visites, sous forme encadree. L'encadrement est impose
    // par l'expansion paresseuse : selon qu'un noeud a ete developpe en tant
    // que feuille (defaut de table) ou traverse en creant ses enfants au vol
    // (succes de table), il a recu ou non une visite propre.
    const uint64_t visites = node->visit_count >= 0
        ? static_cast<uint64_t>(node->visit_count)
        : 0;
    if (node->visit_count >= 0 &&
        (visites < somme_visites || visites > somme_visites + 1)) {
        ajouter_violation(rapport,
            "visites hors encadrement : noeud " + std::to_string(visites) +
            ", enfants " + std::to_string(somme_visites));
    }

    // Priors sommant a 1 : expand_node_single divise par sum_legal.
    if (std::fabs(somme_priors - 1.0f) > 1e-3f) {
        ajouter_violation(rapport,
            "priors ne sommant pas a 1 : " + std::to_string(somme_priors));
    }

    for (const auto& pair : node->children) {
        visiter(pair.second.get(), profondeur + 1, rapport);
    }
}

TreeReport MCTS::inspect_tree() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return inspect_tree(m_analysis_root.get());
}

TreeReport MCTS::inspect_tree(const MCTSNode* root) const {
    TreeReport rapport;
    if (root == nullptr) return rapport;

    rapport.root_visits = root->visit_count >= 0
        ? static_cast<uint64_t>(root->visit_count)
        : 0;
    visiter(root, 0, rapport);
    return rapport;
}
