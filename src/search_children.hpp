#pragma once

#include "evaluation_cache.hpp"
#include "mcts.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// Construit les enfants d'un noeud depuis une entree de table, avec les priors
// normalises stockes. Partage par l'expansion paresseuse de select_leaf et par
// l'expansion explicite de racine, pour que les deux produisent exactement les
// memes enfants.
inline std::vector<std::pair<int, std::unique_ptr<MCTSNode>>>
make_children_from_probe(MCTSNode* parent, const TTProbe& probe) {
    if (probe.policy_size <= 0) {
        throw std::logic_error("hit TT sans politique legale");
    }

    float sum_legal = 0.0f;
    for (int i = 0; i < probe.policy_size; ++i) {
        sum_legal += probe.legal_policy[i].second;
    }

    std::vector<std::pair<int, std::unique_ptr<MCTSNode>>> children;
    children.reserve(static_cast<std::size_t>(probe.policy_size));
    const float uniform = 1.0f / static_cast<float>(probe.policy_size);
    for (int i = 0; i < probe.policy_size; ++i) {
        const int move_index = probe.legal_policy[i].first;
        const float prior = sum_legal > 0.0f
            ? probe.legal_policy[i].second / sum_legal
            : uniform;
        children.emplace_back(
            move_index,
            std::make_unique<MCTSNode>(prior, move_index, parent));
    }
    return children;
}
