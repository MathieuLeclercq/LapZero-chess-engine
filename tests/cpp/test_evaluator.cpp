#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "mcts_test_access.hpp"
#include "test_support.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void test_single_evaluation_uses_batch_contract() {
    ControlledEvaluator evaluator;
    evaluator.output_value = 0.375f;
    bool hook_called = false;
    evaluator.before_evaluate = [&]() { hook_called = true; };

    std::vector<float> input(119 * 64, 0.0f);
    std::vector<float> policy;
    float value = 0.0f;
    evaluator.evaluate(input, policy, value);

    require_test(evaluator.batch_sizes == std::vector<int>{1},
                 "single evaluation did not use a batch of one");
    require_test(hook_called, "evaluation hook was not called");
    require_test(policy.size() == 4672, "single policy has wrong size");
    require_test(value == 0.375f, "single value was not forwarded");
}

void test_mcts_accepts_an_injected_evaluator() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.setStartupPieces();

    mcts.step_analysis(board, 17, 1.4f, 8);
    const TreeReport report = mcts.inspect_tree();

    require_test(report.violations == 0, "invalid reference tree");
    require_test(!evaluator.batch_sizes.empty(), "network was not called");
}

void test_controlled_failure_is_observable() {
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 1;
    std::vector<float> input(119 * 64, 0.0f);
    std::vector<float> policy;
    float value = 0.0f;

    bool threw = false;
    try {
        evaluator.evaluate(input, policy, value);
    }
    catch (const std::runtime_error&) {
        threw = true;
    }
    require_test(threw, "controlled failure did not propagate");
}

void test_scalar_output_is_validated_before_use() {
    const ControlledEvaluator::Corruption corruptions[] = {
        ControlledEvaluator::Corruption::Empty,
        ControlledEvaluator::Corruption::Oversize,
        ControlledEvaluator::Corruption::NonFinite,
    };

    for (auto corruption : corruptions) {
        for (bool sur_les_valeurs : {false, true}) {
            ControlledEvaluator evaluator;
            evaluator.corrupt_call = 1;
            if (sur_les_valeurs) evaluator.values_corruption = corruption;
            else evaluator.policies_corruption = corruption;

            std::vector<float> input(119 * 64, 0.0f);
            std::vector<float> policy;
            float value = 0.0f;
            std::string message;
            bool threw = false;
            try {
                evaluator.evaluate(input, policy, value);
            }
            catch (const std::exception& error) {
                threw = true;
                message = error.what();
            }
            require_test(threw,
                         "invalid scalar evaluator output was accepted");
            require_test(!message.empty(),
                         "the rejection carries no message");
            require_test(message.find("evaluateur") != std::string::npos,
                         "the rejection message does not name the evaluator");
        }
    }
}

void test_root_expansion_rejects_invalid_output_and_recovers() {
    ControlledEvaluator evaluator;
    evaluator.corrupt_call = 1;
    evaluator.policies_corruption =
        ControlledEvaluator::Corruption::NonFinite;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.setStartupPieces();

    bool threw = false;
    try {
        mcts.step_analysis(board, 0, 1.4f, 0, 1);
    }
    catch (const std::exception&) {
        threw = true;
    }
    require_test(threw, "root expansion accepted a non-finite policy");
    const TreeReport after = mcts.inspect_tree();
    require_test(after.violations == 0 && after.en_vol == 0
                     && after.pending == 0,
                 "rejected root expansion left the tree dirty");

    // Aucune entree fautive ne doit empoisonner le cache : un evaluateur sain
    // reevalue la meme position et la recherche repart.
    evaluator.corrupt_call = 0;
    mcts.step_analysis(board, 3, 1.4f, 0, 1);
    const TreeReport recovered = mcts.inspect_tree();
    require_test(recovered.root_visits == 3,
                 "search did not recover after a rejected expansion");
    require_test(recovered.violations == 0,
                 "recovered tree violates invariants");
}

void test_rejected_output_does_not_poison_the_cache() {
    ControlledEvaluator evaluator;
    evaluator.corrupt_call = 1;
    evaluator.values_corruption =
        ControlledEvaluator::Corruption::NonFinite;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.setStartupPieces();
    const EvaluationCacheKey key = MCTSTestAccess::key_for(mcts, board);

    bool threw = false;
    try {
        mcts.step_analysis(board, 0, 1.4f, 0, 1);
    }
    catch (const std::exception&) {
        threw = true;
    }
    require_test(threw, "invalid output was accepted");
    require_test(
        MCTSTestAccess::probe_cache(mcts, key).status == TTProbeStatus::MISS,
        "the rejected key was cached");
    require_test(mcts.inspect_tree().violations == 0,
                 "rejected expansion left the tree dirty");

    // Le meme faux evaluateur, assaini, doit reevaluer la position et remplir
    // le cache avec une entree finie. Une simulation est necessaire : a budget
    // nul, step_analysis ne reexpand pas une racine deja creee.
    evaluator.corrupt_call = 0;
    mcts.step_analysis(board, 1, 1.4f, 0, 1);

    const TTProbe probe = MCTSTestAccess::probe_cache(mcts, key);
    require_test(probe.status == TTProbeStatus::HIT,
                 "a valid evaluation did not fill the cache");
    require_test(std::isfinite(probe.value), "non-finite cached value");
    require_test(probe.policy_size > 0, "empty cached policy");
    for (int i = 0; i < probe.policy_size; ++i) {
        require_test(std::isfinite(probe.legal_policy[i].second),
                     "non-finite cached policy entry");
    }
    require_test(mcts.inspect_tree().violations == 0,
                 "recovered tree violates the invariants");
    require_test(evaluator.batch_sizes.size() == 2,
                 "the position was not evaluated exactly twice");
}

}  // namespace

int main() {
    try {
        test_single_evaluation_uses_batch_contract();
        test_mcts_accepts_an_injected_evaluator();
        test_controlled_failure_is_observable();
        test_scalar_output_is_validated_before_use();
        test_root_expansion_rejects_invalid_output_and_recovers();
        test_rejected_output_does_not_poison_the_cache();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
