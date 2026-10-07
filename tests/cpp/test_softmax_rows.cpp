#include "softmax.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int POLICY_SIZE = 4672;

// Reference sequentielle : meme ordre d'operations que le chemin deploye.
void softmax_sequentiel(const float* logits, float* probabilities,
                        int rows, int columns) {
    for (int row = 0; row < rows; ++row) {
        const float* source =
            logits + static_cast<std::ptrdiff_t>(row) * columns;
        float* target =
            probabilities + static_cast<std::ptrdiff_t>(row) * columns;
        const float max_logit = *std::max_element(source, source + columns);
        float sum_exp = 0.0f;
        for (int i = 0; i < columns; ++i) {
            const float e = std::exp(source[i] - max_logit);
            target[i] = e;
            sum_exp += e;
        }
        const float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < columns; ++i) {
            target[i] *= inv_sum;
        }
    }
}

void verifier_identique(const std::vector<float>& logits, int rows,
                        int columns) {
    std::vector<float> attendu(
        static_cast<std::size_t>(rows) * columns, 0.0f);
    std::vector<float> obtenu(attendu.size(), 0.0f);
    softmax_sequentiel(logits.data(), attendu.data(), rows, columns);
    softmax_rows(logits.data(), obtenu.data(), rows, columns);

    for (std::size_t i = 0; i < attendu.size(); ++i) {
        require_test(attendu[i] == obtenu[i],
                     "le softmax parallele n'est pas identique au sequentiel");
    }

    // Le mode deploye, autorisation de parallelisation a faux, doit etre
    // identique lui aussi : le desactiver ne change aucune valeur.
    std::vector<float> deploye(attendu.size(), 0.0f);
    softmax_rows(logits.data(), deploye.data(), rows, columns, false);
    for (std::size_t i = 0; i < attendu.size(); ++i) {
        require_test(attendu[i] == deploye[i],
                     "le softmax deploye n'est pas identique au sequentiel");
    }
    for (int row = 0; row < rows; ++row) {
        float somme = 0.0f;
        for (int i = 0; i < columns; ++i) {
            somme += obtenu[static_cast<std::size_t>(row) * columns + i];
        }
        require_test(std::fabs(somme - 1.0f) < 1e-4f,
                     "une ligne de softmax ne somme pas a un");
    }
}

std::vector<float> logits_aleatoires(std::mt19937& rng, int rows,
                                     int columns, float echelle) {
    std::uniform_real_distribution<float> dis(-echelle, echelle);
    std::vector<float> logits(static_cast<std::size_t>(rows) * columns);
    for (float& value : logits) value = dis(rng);
    return logits;
}

void test_identique_en_sequentiel_et_parallele() {
    std::mt19937 rng(12345);
    for (const int rows : {1, 7, 15, 16, 17, 64}) {
        const std::vector<float> logits =
            logits_aleatoires(rng, rows, POLICY_SIZE, 5.0f);
        verifier_identique(logits, rows, POLICY_SIZE);
    }
}

void test_identique_sur_entrees_extremes() {
    std::mt19937 rng(999);
    std::vector<float> logits = logits_aleatoires(rng, 32, POLICY_SIZE, 2.0f);
    // Un logit ecrasant : toutes les exponentielles des autres cases
    // sous-debordent vers zero. Le resultat doit rester identique.
    logits[5 * POLICY_SIZE + 17] = 300.0f;
    verifier_identique(logits, 32, POLICY_SIZE);

    // Deux maxima egaux et des logits tres negatifs.
    std::vector<float> plats(16 * POLICY_SIZE, -500.0f);
    for (int row = 0; row < 16; ++row) {
        plats[static_cast<std::size_t>(row) * POLICY_SIZE + 3] = 0.0f;
        plats[static_cast<std::size_t>(row) * POLICY_SIZE + 9] = 0.0f;
    }
    verifier_identique(plats, 16, POLICY_SIZE);
}

void test_formes_particulieres() {
    std::mt19937 rng(2026);
    for (const int columns : {1, 2, 3, 64}) {
        const std::vector<float> logits =
            logits_aleatoires(rng, 20, columns, 4.0f);
        verifier_identique(logits, 20, columns);
    }
}

void test_deterministe_entre_appels() {
    std::mt19937 rng(77);
    const std::vector<float> logits =
        logits_aleatoires(rng, 32, POLICY_SIZE, 6.0f);
    std::vector<float> premier(logits.size(), 0.0f);
    std::vector<float> second(logits.size(), 0.0f);
    softmax_rows(logits.data(), premier.data(), 32, POLICY_SIZE);
    softmax_rows(logits.data(), second.data(), 32, POLICY_SIZE);
    require_test(premier == second,
                 "deux softmax paralleles ne donnent pas le meme resultat");
}

}  // namespace

int main() {
    try {
        test_identique_en_sequentiel_et_parallele();
        test_identique_sur_entrees_extremes();
        test_formes_particulieres();
        test_deterministe_entre_appels();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
