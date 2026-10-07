#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "onnx_evaluator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int GAME_COUNT = 8;
constexpr int INPUT_SIZE = 119 * 64;
constexpr int POLICY_SIZE = 4672;

struct Options {
    bool fake = false;
    bool gpu = false;
    std::string model;
    int repetitions = 30;
    int rounds = 100;
};

struct Result {
    std::uint64_t collection_ns = 0;
    std::uint64_t processing_ns = 0;
    std::uint64_t nn_calls = 0;
    std::uint64_t nn_batches = 0;
    SearchCounters counters;
    std::vector<std::pair<int, int>> first_root_visits;
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
        else if (arg == "--model" && i + 1 < argc) {
            options.model = argv[++i];
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

void warm_up(Evaluator& evaluator) {
    std::vector<float> input(GAME_COUNT * INPUT_SIZE, 0.0f);
    std::vector<float> policies;
    std::vector<float> values;
    evaluator.evaluate_batch(input, policies, values, GAME_COUNT);
}

Result run_repetition(Evaluator& evaluator, int rounds) {
    MCTS mcts(&evaluator, 8192, 0);
    std::vector<Chessboard> boards(GAME_COUNT);
    std::vector<std::unique_ptr<MCTSNode>> roots;
    roots.reserve(GAME_COUNT);

    for (int i = 0; i < GAME_COUNT; ++i) {
        boards[i].setStartupPieces();
        roots.push_back(std::make_unique<MCTSNode>(0.0f));
        mcts.expand_node_single(roots.back().get(), boards[i]);
    }
    mcts.reset_counters();

    std::vector<MCTSNode*> leaves(GAME_COUNT, nullptr);
    std::vector<int> moves_played(GAME_COUNT, 0);
    std::vector<PathReservation> reservations(GAME_COUNT);
    std::vector<std::vector<float>> tensors(
        GAME_COUNT, std::vector<float>(INPUT_SIZE));
    std::vector<float> batch_input;
    batch_input.reserve(GAME_COUNT * INPUT_SIZE);
    std::vector<float> policies;
    std::vector<float> values;

    Result result;
    for (int round = 0; round < rounds; ++round) {
        std::fill(leaves.begin(), leaves.end(), nullptr);
        std::fill(moves_played.begin(), moves_played.end(), 0);

        const auto collection_start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(GAME_COUNT) schedule(static)
        for (int game = 0; game < GAME_COUNT; ++game) {
            leaves[game] = mcts.advance_to_leaf(
                roots[game].get(), boards[game], 1.4f,
                moves_played[game], reservations[game]);
            if (leaves[game] != nullptr) {
                boards[game].getAlphaZeroTensor(tensors[game]);
            }
        }
        result.collection_ns += elapsed_ns(collection_start);

        const auto processing_start = std::chrono::steady_clock::now();
        batch_input.clear();
        std::vector<int> active_games;
        active_games.reserve(GAME_COUNT);
        for (int game = 0; game < GAME_COUNT; ++game) {
            if (leaves[game] == nullptr) {
                continue;
            }
            active_games.push_back(game);
            batch_input.insert(batch_input.end(), tensors[game].begin(),
                               tensors[game].end());
        }

        if (!active_games.empty()) {
            result.nn_calls += active_games.size();
            result.nn_batches += 1;
            evaluator.evaluate_batch(batch_input, policies, values,
                                     static_cast<int>(active_games.size()));
            for (std::size_t batch_index = 0;
                 batch_index < active_games.size(); ++batch_index) {
                const int game = active_games[batch_index];
                mcts.expand_and_backup(
                    leaves[game], boards[game],
                    policies.data() + batch_index * POLICY_SIZE,
                    values[batch_index], reservations[game]);
                for (int move = 0; move < moves_played[game]; ++move) {
                    boards[game].undoMove();
                }
            }
        }
        result.processing_ns += elapsed_ns(processing_start);
    }

    result.counters = mcts.get_counters();
    if (roots.front()->visit_count != rounds) {
        throw std::runtime_error("budget de visites incorrect sur la racine");
    }
    for (const auto& child : roots.front()->children) {
        result.first_root_visits.push_back(
            {child.first, child.second->visit_count});
    }
    std::sort(result.first_root_visits.begin(),
              result.first_root_visits.end());
    return result;
}

void print_json(const Options& options, const std::vector<Result>& results) {
    std::cout << "{\n"
              << "  \"mode\": \"" << (options.fake ? "fake" : "model")
              << "\",\n"
              << "  \"gpu\": " << (options.gpu ? "true" : "false") << ",\n"
              << "  \"games\": " << GAME_COUNT << ",\n"
              << "  \"rounds\": " << options.rounds << ",\n"
              << "  \"repetitions\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const Result& result = results[i];
        const SearchCounters& c = result.counters;
        std::cout << "    {\"index\": " << i
                  << ", \"collection_ns\": " << result.collection_ns
                  << ", \"processing_ns\": " << result.processing_ns
                  << ", \"nn_calls\": " << result.nn_calls
                  << ", \"nn_batches\": " << result.nn_batches
                  << ", \"tt_hits\": " << c.tt_hits
                  << ", \"tt_misses\": " << c.tt_misses
                  << ", \"tt_position_matches\": "
                  << c.tt_position_matches
                  << ", \"tt_rule50_rejects\": " << c.tt_rule50_rejects
                  << ", \"tt_context_rejects\": " << c.tt_context_rejects
                  << ", \"tt_history_rejects\": " << c.tt_history_rejects
                  << ", \"terminal_hits\": " << c.terminal_hits
                  << "}";
        if (i + 1 != results.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }
    std::cout << "  ],\n  \"reference_visits\": [";
    if (!results.empty()) {
        const auto& visits = results.front().first_root_visits;
        for (std::size_t i = 0; i < visits.size(); ++i) {
            std::cout << "[" << visits[i].first << ", "
                      << visits[i].second << "]";
            if (i + 1 != visits.size()) {
                std::cout << ", ";
            }
        }
    }
    std::cout << "]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::unique_ptr<Evaluator> evaluator;
        if (options.fake) {
            evaluator = std::make_unique<ControlledEvaluator>();
        }
        else {
            evaluator = std::make_unique<ONNXEvaluator>(
                options.model, options.gpu);
        }

        warm_up(*evaluator);
        std::vector<Result> results;
        results.reserve(options.repetitions);
        for (int repetition = 0; repetition < options.repetitions;
             ++repetition) {
            results.push_back(run_repetition(*evaluator, options.rounds));
        }
        print_json(options, results);
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
