#pragma once

#include "evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

class ControlledEvaluator final : public Evaluator {
public:
    // Corruption ciblee d'une sortie, pour prouver que le consommateur valide
    // avant tout acces par indice.
    enum class Corruption {
        None,
        Empty,
        Oversize,
        NonFinite,
    };

    std::vector<int> batch_sizes;
    EvaluatorTotals totals;
    int fail_on_call = 0;
    int truncate_policy_on_call = 0;
    int truncate_values_on_call = 0;
    float output_value = 0.0f;
    std::function<void()> before_evaluate;

    // Applique une corruption aux politiques, aux valeurs ou aux deux, au
    // numero d'appel donne. corrupt_line = -1 vise la derniere ligne du lot.
    int corrupt_call = 0;
    Corruption policies_corruption = Corruption::None;
    Corruption values_corruption = Corruption::None;
    int corrupt_line = -1;

    void evaluate_batch(const std::vector<float>& input,
                        std::vector<float>& policies,
                        std::vector<float>& values,
                        int batch_size) override {
        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size doit etre positif");
        }
        const std::size_t expected =
            static_cast<std::size_t>(batch_size) * 119 * 64;
        if (input.size() != expected) {
            throw std::invalid_argument("taille du tenseur inattendue");
        }

        batch_sizes.push_back(batch_size);
        totals.run_calls++;
        totals.evaluated_rows += static_cast<std::uint64_t>(batch_size);
        if (before_evaluate) {
            before_evaluate();
        }
        if (fail_on_call > 0
            && static_cast<int>(batch_sizes.size()) == fail_on_call) {
            throw std::runtime_error("echec controle de l evaluateur");
        }

        policies.assign(static_cast<std::size_t>(batch_size) * POLICY_SIZE,
                        1.0f / static_cast<float>(POLICY_SIZE));
        values.assign(batch_size, output_value);
        const int call = static_cast<int>(batch_sizes.size());
        if (truncate_policy_on_call == call && !policies.empty()) {
            policies.pop_back();
        }
        if (truncate_values_on_call == call && !values.empty()) {
            values.pop_back();
        }
        appliquer_corruption(policies, values, batch_size, call);
    }

    EvaluatorTotals diagnostic_totals() const override { return totals; }

private:
    std::size_t ligne_visee(int batch_size) const {
        const int line = corrupt_line >= 0 ? corrupt_line : batch_size - 1;
        return static_cast<std::size_t>(line);
    }

    void appliquer_corruption(std::vector<float>& policies,
                              std::vector<float>& values,
                              int batch_size, int call) {
        if (call != corrupt_call) return;
        if (policies_corruption == Corruption::Empty) {
            policies.clear();
        }
        else if (policies_corruption == Corruption::Oversize) {
            policies.push_back(0.0f);
        }
        else if (policies_corruption == Corruption::NonFinite) {
            const std::size_t index =
                ligne_visee(batch_size) * POLICY_SIZE + 17;
            policies[index] = std::numeric_limits<float>::quiet_NaN();
        }

        if (values_corruption == Corruption::Empty) {
            values.clear();
        }
        else if (values_corruption == Corruption::Oversize) {
            values.push_back(0.0f);
        }
        else if (values_corruption == Corruption::NonFinite) {
            values[ligne_visee(batch_size)] =
                std::numeric_limits<float>::quiet_NaN();
        }
    }
};
