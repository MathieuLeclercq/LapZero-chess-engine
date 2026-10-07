#pragma once

#include "evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Evaluateur de test qui identifie chaque ligne par le tenseur recu, jamais par
// l'ordre suppose des workers. Chaque tenseur distinct recoit une valeur et une
// politique distinctes : un decalage entre lignes et feuilles devient visible.
//
// En mode sentinelles de padding, la deuxieme presentation d'un meme tenseur
// (donc une ligne dupliquee par le lot de forme fixe) recoit une reponse
// volontairement differente. Sa consommation accidentelle par une feuille
// reelle se voit alors immediatement. Ce mode ne sert qu'a la detection.
class DiscriminatingEvaluator final : public Evaluator {
public:
    struct Reponse {
        float value = 0.0f;
        std::vector<float> policy;
    };

    // Sentinelles de padding : une ligne repetee (meme tenseur) recoit une
    // reponse differente, pour detecter sa consommation accidentelle.
    bool sentinelles_padding = false;

    std::vector<int> batch_sizes;

    static std::string signature(const std::vector<float>& tensor) {
        return std::string(
            reinterpret_cast<const char*>(tensor.data()),
            tensor.size() * sizeof(float));
    }

    bool a_une_reponse(const std::vector<float>& tensor) const {
        return m_reponses.count(signature(tensor)) > 0;
    }

    const Reponse& reponse_pour(const std::vector<float>& tensor) const {
        return m_reponses.at(signature(tensor));
    }

    void evaluate_batch(const std::vector<float>& input,
                        std::vector<float>& policies,
                        std::vector<float>& values,
                        int batch_size) override {
        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size doit etre positif");
        }
        const std::size_t attendu =
            static_cast<std::size_t>(batch_size) * 119 * 64;
        if (input.size() != attendu) {
            throw std::invalid_argument("taille du tenseur inattendue");
        }
        batch_sizes.push_back(batch_size);

        policies.assign(
            static_cast<std::size_t>(batch_size) * POLICY_SIZE, 0.0f);
        values.assign(batch_size, 0.0f);

        std::unordered_map<std::string, int> occurrences;
        for (int ligne = 0; ligne < batch_size; ++ligne) {
            const float* debut =
                input.data() + static_cast<std::size_t>(ligne) * 119 * 64;
            const std::vector<float> tensor(debut, debut + 119 * 64);
            const std::string sig = signature(tensor);

            if (m_reponses.count(sig) == 0) {
                m_reponses.emplace(sig, creer_reponse());
            }
            Reponse reponse = m_reponses.at(sig);
            const int occurrence = occurrences[sig]++;
            if (sentinelles_padding && occurrence > 0) {
                reponse = reponse_sentinelle(occurrence);
            }

            std::copy(reponse.policy.begin(), reponse.policy.end(),
                      policies.begin()
                          + static_cast<std::size_t>(ligne) * POLICY_SIZE);
            values[ligne] = reponse.value;
        }
    }

private:
    Reponse creer_reponse() const {
        static constexpr float VALEURS[4] = {
            -0.6f, -0.2f, 0.3f, 0.7f};
        const int rang = static_cast<int>(m_reponses.size());

        Reponse reponse;
        reponse.value = VALEURS[rang % 4];
        // Toutes les cases sont positives : la politique a toujours de la masse
        // sur les coups legaux, quelle que soit la position du noeud. Le motif
        // depend du rang, donc deux rang differents ne donnent jamais les memes
        // priors sur les memes enfants.
        reponse.policy.resize(POLICY_SIZE);
        for (int i = 0; i < POLICY_SIZE; ++i) {
            reponse.policy[static_cast<std::size_t>(i)] =
                1.0f + 0.25f * static_cast<float>((i * (rang + 3)) % 7);
        }
        return reponse;
    }

    Reponse reponse_sentinelle(int occurrence) const {
        Reponse reponse;
        reponse.value = 100.0f + static_cast<float>(occurrence);
        reponse.policy.assign(POLICY_SIZE, 0.0f);
        reponse.policy[0] = 1.0f;
        return reponse;
    }

    std::unordered_map<std::string, Reponse> m_reponses;
};
