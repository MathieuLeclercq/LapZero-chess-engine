// Banc du cout de l'evaluateur hors arbre.
//
// Mesure la courbe du lot (millisecondes par appel et par position) sur GPU et
// CPU, separe session->Run du softmax C++, et propose un mode --variable qui
// alterne les lots 1 a 8 comme le fait la recherche.
//
// Voir docs/superpowers/specs/2026-09-19-cout-calcul-cpu-gpu.md, tache 1.

#include "controlled_evaluator.hpp"
#include "evaluator.hpp"
#include "onnx_evaluator.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int INPUT_SIZE = 119 * 64;
constexpr int VARIABLE_MAX_BATCH = 8;

struct Options {
    bool fake = false;
    bool gpu = false;
    bool variable = false;
    std::string model;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int repetitions = 30;
    int rounds = 20;
};

struct Mesure {
    int batch = 0;
    int repetition = 0;
    int calls = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t run_ns = 0;
    std::uint64_t softmax_ns = 0;
};

int parse_positive(const char* text, const char* option) {
    const int value = std::atoi(text);
    if (value <= 0) {
        throw std::invalid_argument(std::string(option) + " doit etre positif");
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fake") {
            options.fake = true;
        }
        else if (arg == "--gpu") {
            options.gpu = true;
        }
        else if (arg == "--variable") {
            options.variable = true;
        }
        else if (arg == "--model" && i + 1 < argc) {
            options.model = argv[++i];
        }
        else if (arg == "--batches") {
            options.batches.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                options.batches.push_back(
                    parse_positive(argv[++i], "batches"));
            }
            if (options.batches.empty()) {
                throw std::invalid_argument("--batches attend au moins une valeur");
            }
        }
        else if (arg == "--repetitions" && i + 1 < argc) {
            options.repetitions = parse_positive(argv[++i], "repetitions");
        }
        else if (arg == "--rounds" && i + 1 < argc) {
            options.rounds = parse_positive(argv[++i], "rounds");
        }
        else {
            throw std::invalid_argument("option inconnue ou incomplete : " + arg);
        }
    }

    if (options.fake == !options.model.empty()) {
        throw std::invalid_argument(
            "utiliser exactement un de --fake ou --model PATH");
    }
    if (options.fake && options.gpu) {
        throw std::invalid_argument("--gpu exige --model");
    }
    return options;
}

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
}

std::vector<float> tenseur_aleatoire(std::mt19937& rng, int batch) {
    std::vector<float> input(
        static_cast<std::size_t>(batch) * INPUT_SIZE);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (float& value : input) {
        value = dis(rng);
    }
    return input;
}

Mesure mesurer_lot(Evaluator& evaluator, ONNXEvaluator* onnx,
                   std::mt19937& rng, int batch, int rounds,
                   int repetition) {
    std::vector<float> input = tenseur_aleatoire(rng, batch);
    std::vector<float> policies;
    std::vector<float> values;

    const auto debut = std::chrono::steady_clock::now();
    for (int call = 0; call < rounds; ++call) {
        evaluator.evaluate_batch(input, policies, values, batch);
    }
    const auto fin = std::chrono::steady_clock::now();

    Mesure mesure;
    mesure.batch = batch;
    mesure.repetition = repetition;
    mesure.calls = rounds;
    mesure.wall_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            fin - debut).count());
    if (onnx != nullptr) {
        const EvaluatorTiming timing = onnx->get_last_timing();
        mesure.run_ns = timing.run_ns;
        mesure.softmax_ns = timing.softmax_ns;
    }
    return mesure;
}

std::vector<Mesure> mesurer_variable(Evaluator& evaluator,
                                     ONNXEvaluator* onnx,
                                     std::mt19937& rng, int rounds,
                                     int repetition) {
    std::vector<std::vector<float>> inputs;
    inputs.reserve(VARIABLE_MAX_BATCH + 1);
    inputs.emplace_back();
    for (int batch = 1; batch <= VARIABLE_MAX_BATCH; ++batch) {
        inputs.push_back(tenseur_aleatoire(rng, batch));
    }

    std::vector<std::uint64_t> wall(VARIABLE_MAX_BATCH + 1, 0);
    std::vector<std::uint64_t> run(VARIABLE_MAX_BATCH + 1, 0);
    std::vector<std::uint64_t> softmax(VARIABLE_MAX_BATCH + 1, 0);
    std::vector<float> policies;
    std::vector<float> values;

    for (int cycle = 0; cycle < rounds; ++cycle) {
        for (int batch = 1; batch <= VARIABLE_MAX_BATCH; ++batch) {
            const auto debut = std::chrono::steady_clock::now();
            evaluator.evaluate_batch(inputs[batch], policies, values, batch);
            wall[batch] += elapsed_ns(debut);
            if (onnx != nullptr) {
                const EvaluatorTiming timing = onnx->get_last_timing();
                run[batch] += timing.run_ns;
                softmax[batch] += timing.softmax_ns;
            }
        }
    }

    std::vector<Mesure> mesures;
    for (int batch = 1; batch <= VARIABLE_MAX_BATCH; ++batch) {
        Mesure mesure;
        mesure.batch = batch;
        mesure.repetition = repetition;
        mesure.calls = rounds;
        mesure.wall_ns = wall[batch];
        mesure.run_ns = run[batch];
        mesure.softmax_ns = softmax[batch];
        mesures.push_back(mesure);
    }
    return mesures;
}

void print_json(const Options& options, const std::vector<Mesure>& mesures) {
    std::cout << "{\n"
              << "  \"mode\": \"" << (options.fake ? "fake" : "model")
              << "\",\n"
              << "  \"gpu\": " << (options.gpu ? "true" : "false") << ",\n"
              << "  \"variable\": " << (options.variable ? "true" : "false")
              << ",\n"
              << "  \"repetitions\": " << options.repetitions << ",\n"
              << "  \"rounds\": " << options.rounds << ",\n"
              << "  \"measurements\": [\n";
    for (std::size_t i = 0; i < mesures.size(); ++i) {
        const Mesure& mesure = mesures[i];
        const double ms_appel = mesure.wall_ns / 1e6 / mesure.calls;
        const double ms_position = ms_appel / mesure.batch;
        std::cout << "    {\"batch\": " << mesure.batch
                  << ", \"repetition\": " << mesure.repetition
                  << ", \"calls\": " << mesure.calls
                  << ", \"wall_ns\": " << mesure.wall_ns
                  << ", \"ms_appel\": " << std::fixed
                  << std::setprecision(4) << ms_appel
                  << ", \"ms_position\": " << std::setprecision(6)
                  << ms_position
                  << ", \"run_ns\": " << mesure.run_ns
                  << ", \"softmax_ns\": " << mesure.softmax_ns << "}";
        if (i + 1 != mesures.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }
    std::cout << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::unique_ptr<Evaluator> evaluator;
        ONNXEvaluator* onnx = nullptr;
        if (options.fake) {
            evaluator = std::make_unique<ControlledEvaluator>();
        }
        else {
            auto instance = std::make_unique<ONNXEvaluator>(
                options.model, options.gpu);
            instance->set_timing_enabled(true);
            onnx = instance.get();
            evaluator = std::move(instance);
        }

        std::mt19937 rng(12345);
        const std::vector<int> chauffe = options.variable
            ? std::vector<int>{1, 2, 4, 8}
            : options.batches;
        for (int batch : chauffe) {
            std::vector<float> input = tenseur_aleatoire(rng, batch);
            std::vector<float> policies;
            std::vector<float> values;
            evaluator->evaluate_batch(input, policies, values, batch);
        }

        std::vector<Mesure> mesures;
        for (int repetition = 0; repetition < options.repetitions;
             ++repetition) {
            if (options.variable) {
                const std::vector<Mesure> lot = mesurer_variable(
                    *evaluator, onnx, rng, options.rounds, repetition);
                mesures.insert(mesures.end(), lot.begin(), lot.end());
            }
            else {
                for (int batch : options.batches) {
                    mesures.push_back(mesurer_lot(
                        *evaluator, onnx, rng, batch, options.rounds,
                        repetition));
                }
            }
        }

        print_json(options, mesures);
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
