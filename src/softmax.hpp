#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>

// Seuil au-dela duquel les lignes sont reparties entre threads. En dessous,
// une region OpenMP coute plus cher que le calcul (cas du bot UCI, lot de 8).
inline constexpr int SOFTMAX_PARALLEL_MIN_ROWS = 16;

// Softmax par ligne, arithmetiquement identique au chemin sequentiel : chaque
// ligne est independante et conserve l'ordre d'origine de ses operations, donc
// le resultat est bit a bit le meme. La parallelisation porte sur les lignes,
// jamais sur la reduction d'une ligne. Appele depuis une region OpenMP, le
// chemin reste sequentiel pour ne pas imbriquer les equipes.
//
// Le mode parallele est desactive par defaut dans la production self-play :
// mesure du 2026-09-25, a 128 places et machine chargee, une region OpenMP par
// appel coutait plus cher que les exponentielles economisees (163 s de softmax
// contre environ 50 s attendues en sequentiel). Le gain du banc isole, lot
// fixe et machine au repos, ne se transpose pas.
inline void softmax_rows(const float* logits, float* probabilities,
                         int rows, int columns, bool allow_parallel = true) {
    if (rows <= 0 || columns <= 0) return;
    const bool parallel = allow_parallel && rows >= SOFTMAX_PARALLEL_MIN_ROWS
        && !omp_in_parallel() && omp_get_max_threads() > 1;
#pragma omp parallel for schedule(static) if(parallel)
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
