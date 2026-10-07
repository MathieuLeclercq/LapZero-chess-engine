#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "test_support.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

std::unique_ptr<MCTSNode> child_of(MCTSNode& parent, float prior = 1.0f) {
    return std::make_unique<MCTSNode>(prior, 8, &parent);
}

TreeReport inspect(MCTSNode& root) {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 16, 0);
    return mcts.inspect_tree(&root);
}

void test_pending_is_reported_as_a_rest_violation() {
    MCTSNode root(0.0f);
    root.state.store(NodeState::Pending);

    const TreeReport report = inspect(root);

    require_test(report.pending == 1, "Pending node was not counted");
    require_test(report.violations > 0, "Pending at rest was accepted");
}

void test_publication_state_matches_children() {
    MCTSNode unexpanded(0.0f);
    unexpanded.children.emplace_back(8, child_of(unexpanded));
    require_test(inspect(unexpanded).violations > 0,
                 "Unexpanded node with children was accepted");

    MCTSNode expanded(0.0f);
    expanded.state.store(NodeState::Expanded);
    require_test(inspect(expanded).violations > 0,
                 "Expanded node without children was accepted");

    MCTSNode terminal(0.0f);
    terminal.state.store(NodeState::Terminal);
    terminal.children.emplace_back(8, child_of(terminal));
    require_test(inspect(terminal).violations > 0,
                 "Terminal node with children was accepted");
}

void test_non_finite_statistics_are_rejected() {
    MCTSNode root(0.0f);
    root.state.store(NodeState::Expanded);
    root.children.emplace_back(
        8, child_of(root, std::numeric_limits<float>::quiet_NaN()));

    require_test(inspect(root).violations > 0,
                 "non-finite prior was accepted");
}

void test_root_visits_are_exposed() {
    MCTSNode root(0.0f);
    root.visit_count = 17;

    const TreeReport report = inspect(root);

    require_test(report.root_visits == 17, "root visits were not exposed");
}

void test_in_flight_is_reported_as_a_rest_violation() {
    MCTSNode root(0.0f);
    PathReservation reservation;
    reservation.reserve(&root, 3);

    const TreeReport report = inspect(root);

    reservation.release();
    require_test(report.en_vol == 1, "n_in_flight node was not counted in en_vol");
    require_test(report.violations > 0, "n_in_flight at rest was accepted");
}

void test_released_in_flight_is_clean_again() {
    MCTSNode root(0.0f);
    PathReservation reservation;
    reservation.reserve(&root, 1);
    reservation.release();

    const TreeReport report = inspect(root);

    require_test(report.en_vol == 0, "released node is still counted in flight");
    require_test(report.violations == 0, "released node was flagged as a leak");
}

void test_a_clean_tree_is_not_flagged() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 16, 0);

    const TreeReport vide = mcts.inspect_tree();
    require_test(vide.nodes == 0 && vide.violations == 0,
                 "empty analysis tree was flagged");

    MCTSNode root(0.0f);
    root.state.store(NodeState::Expanded);
    root.children.emplace_back(8, child_of(root, 1.0f));

    const TreeReport propre = mcts.inspect_tree(&root);
    require_test(propre.nodes == 2, "clean tree was not walked");
    require_test(propre.en_vol == 0 && propre.pending == 0,
                 "clean tree reported in-flight or pending nodes");
    require_test(propre.violations == 0, "clean tree was flagged");
}

}  // namespace

int main() {
    try {
        test_pending_is_reported_as_a_rest_violation();
        test_publication_state_matches_children();
        test_non_finite_statistics_are_rejected();
        test_root_visits_are_exposed();
        test_in_flight_is_reported_as_a_rest_violation();
        test_released_in_flight_is_clean_again();
        test_a_clean_tree_is_not_flagged();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
