#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Taille de la politique encodee, commune au contrat de l'evaluateur et a ses
// consommateurs.
inline constexpr int POLICY_SIZE = 4672;

// Chronometrage interne optionnel d'un evaluateur, desactive par defaut. Sert a
// separer session->Run du softmax C++.
struct EvaluatorTiming {
    std::uint64_t run_ns = 0;
    std::uint64_t softmax_ns = 0;
};

// Totaux cumules depuis la construction, lus par l'instrumentation du
// self-play. Les appels et les lignes sont toujours comptes ; les durees ne le
// sont que si le chronometrage interne est actif.
struct EvaluatorTotals {
    std::uint64_t run_calls = 0;
    std::uint64_t evaluated_rows = 0;
    EvaluatorTiming timing;
};

// Valide une sortie d'evaluateur AVANT tout acces par indice ou insertion en
// table. Le lot entier est valide avant d'en consommer le premier element : un
// NaN sur la derniere ligne ne doit pas etre decouvert apres stockage des
// premieres. Une sortie invalide est rejetee, jamais corrigee silencieusement.
inline void validate_network_output(const std::vector<float>& policies,
                                    const std::vector<float>& values,
                                    int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("evaluateur : taille de lot invalide");
    }
    if (policies.size()
        != static_cast<std::size_t>(batch_size) * POLICY_SIZE) {
        throw std::runtime_error(
            "evaluateur : dimensions de politique invalides");
    }
    if (values.size() != static_cast<std::size_t>(batch_size)) {
        throw std::runtime_error(
            "evaluateur : dimensions de valeur invalides");
    }
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("evaluateur : valeur non finie");
        }
    }
    for (const float probability : policies) {
        if (!std::isfinite(probability)) {
            throw std::runtime_error("evaluateur : politique non finie");
        }
    }
}

class Evaluator {
public:
    virtual ~Evaluator() = default;

    virtual void evaluate_batch(const std::vector<float>& input,
                                std::vector<float>& policies,
                                std::vector<float>& values,
                                int batch_size) = 0;

    // Active ou desactive les horloges internes. Les evaluateurs de test qui
    // ne mesurent rien gardent l'implementation vide.
    virtual void set_timing_enabled(bool /*enabled*/) {}

    // Totaux cumules, remis a zero uniquement par le constructeur du
    // evaluateur. Un evaluateur de test rend des compteurs nuls.
    virtual EvaluatorTotals diagnostic_totals() const {
        return EvaluatorTotals{};
    }

    void evaluate(const std::vector<float>& input,
                  std::vector<float>& policy,
                  float& value) {
        std::vector<float> values(1);
        evaluate_batch(input, policy, values, 1);
        validate_network_output(policy, values, 1);
        value = values[0];
    }
};
