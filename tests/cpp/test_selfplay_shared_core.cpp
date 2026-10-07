#include "controlled_evaluator.hpp"
#include "discriminating_evaluator.hpp"
#include "mcts.hpp"
#include "mcts_test_access.hpp"
#include "selfplay_manager.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class SelfPlayTestAccess {
public:
    static MCTSNode* root(SelfPlayManager& manager, int game_idx) {
        return manager.m_roots[static_cast<std::size_t>(game_idx)].get();
    }

    static float pending_epsilon(const SelfPlayManager& manager,
                                 int game_idx) {
        return manager.m_pending_epsilon[static_cast<std::size_t>(game_idx)];
    }

    static void demarrer_slot(SelfPlayManager& manager, int game_idx) {
        manager.demarrer_slot(game_idx);
    }

    // Impose la fin d'une partie apres un nombre de plies connu. Le levier est
    // a zero par defaut et ne sert qu'aux tests de comptabilite.
    static void forcer_fin_apres(SelfPlayManager& manager, int game_idx,
                                 int plies) {
        manager.m_forced_end_plies[static_cast<std::size_t>(game_idx)] = plies;
    }

    static void seed_rng(SelfPlayManager& manager, std::uint32_t graine) {
        manager.m_rng.seed(graine);
    }

    static void reset_game(SelfPlayManager& manager, int game_idx) {
        manager.reset_game(game_idx);
    }

    static void play_best_move(SelfPlayManager& manager, int game_idx) {
        manager.play_best_move(game_idx);
    }

    static int sims_target(const SelfPlayManager& manager, int game_idx) {
        return manager.m_sims_target[static_cast<std::size_t>(game_idx)];
    }

    static bool is_slow_move(const SelfPlayManager& manager, int game_idx) {
        return manager.m_is_slow_move[static_cast<std::size_t>(game_idx)] != 0;
    }

    static bool tactical_boost(const SelfPlayManager& manager, int game_idx) {
        return manager.m_tactical_boost[static_cast<std::size_t>(game_idx)]
            != 0;
    }

    static int plafond_plies() {
        return SelfPlayManager::MAX_SELFPLAY_PLIES;
    }

    static bool longueur_max_atteinte(const SelfPlayManager& manager,
                                      int game_idx) {
        return manager.longueur_max_atteinte(game_idx);
    }

    static int historique(const SelfPlayManager& manager, int game_idx) {
        return static_cast<int>(
            manager.m_boards[static_cast<std::size_t>(game_idx)]
                .getMoveHistory().size());
    }

    static float epsilon_for_move(const SelfPlayManager& manager,
                                  int game_idx) {
        return manager.epsilon_for_move(game_idx);
    }

    static void set_is_slow_move(SelfPlayManager& manager, int game_idx,
                                 bool slow) {
        manager.m_is_slow_move[static_cast<std::size_t>(game_idx)] =
            slow ? 1 : 0;
    }

    static void set_sims_target(SelfPlayManager& manager, int game_idx,
                                int simulations) {
        manager.m_sims_target[static_cast<std::size_t>(game_idx)] =
            simulations;
    }

    static void set_tactical_boost(SelfPlayManager& manager, int game_idx,
                                   bool boost) {
        manager.m_tactical_boost[static_cast<std::size_t>(game_idx)] =
            boost ? 1 : 0;
    }

    static void set_pending_epsilon(SelfPlayManager& manager, int game_idx,
                                    float epsilon) {
        manager.m_pending_epsilon[static_cast<std::size_t>(game_idx)] =
            epsilon;
    }

    static void appliquer_bruit_en_attente(SelfPlayManager& manager,
                                           int game_idx) {
        manager.apply_pending_noise(game_idx);
    }

    // Racine sans enfants, pour prouver l'application differee du bruit.
    static void racine_vide(SelfPlayManager& manager, int game_idx) {
        const std::size_t index = static_cast<std::size_t>(game_idx);
        manager.m_boards[index].setStartupPieces();
        manager.m_roots[index] = std::make_unique<MCTSNode>(0.0f);
    }

    static void developper_racine(SelfPlayManager& manager, int game_idx) {
        manager.expand_root(game_idx);
    }

    static void jouer_uci(SelfPlayManager& manager, int game_idx,
                          const std::string& uci) {
        if (!manager.m_boards[static_cast<std::size_t>(game_idx)]
                 .movePieceUCI(uci)) {
            throw std::runtime_error(
                "jouer_uci : coup refuse " + uci + " dans " +
                manager.m_boards[static_cast<std::size_t>(game_idx)].toFEN());
        }
    }

    static MCTS& mcts(SelfPlayManager& manager) {
        return *manager.m_shared_mcts;
    }

    static std::string fen(SelfPlayManager& manager, int game_idx) {
        return manager.m_boards[static_cast<std::size_t>(game_idx)].toFEN();
    }

    // Prepare une position distincte par partie, puis une racine deja
    // developpee, sans passer par le scheduler.
    static void preparer_position(SelfPlayManager& manager, int game_idx,
                                  const std::string& uci) {
        const std::size_t index = static_cast<std::size_t>(game_idx);
        manager.m_boards[index].setStartupPieces();
        if (!uci.empty()) {
            if (!manager.m_boards[index].movePieceUCI(uci)) {
                throw std::runtime_error("preparer_position : coup invalide");
            }
        }
        manager.m_roots[index] = std::make_unique<MCTSNode>(0.0f);
        manager.m_shared_mcts->expand_node_single(
            manager.m_roots[index].get(), manager.m_boards[index]);
    }

    struct ResultatLot {
        std::vector<MCTSNode*> feuilles;
        std::vector<std::vector<float>> tenseurs;
        std::vector<std::vector<int>> coups_legaux;
    };

    // Monte un lot a partir de feuilles de parties distinctes, comme le font
    // les phases de generation, puis execute la consommation de production.
    static ResultatLot executer_lot_par_partie(
            SelfPlayManager& manager, const std::vector<int>& parties) {
        ResultatLot resultat;
        resultat.feuilles.assign(parties.size(), nullptr);
        resultat.tenseurs.resize(parties.size());
        resultat.coups_legaux.resize(parties.size());

        for (std::size_t j = 0; j < parties.size(); ++j) {
            const int i = parties[j];
            int moves_played = 0;
            PathReservation reservation;
            MCTSNode* leaf = manager.m_shared_mcts->advance_to_leaf(
                manager.m_roots[static_cast<std::size_t>(i)].get(),
                manager.m_boards[static_cast<std::size_t>(i)], 1.4f,
                moves_played, reservation);
            if (leaf == nullptr) continue;

            resultat.feuilles[j] = leaf;
            resultat.coups_legaux[j] =
                manager.m_boards[static_cast<std::size_t>(i)]
                    .getLegalMoveIndices();
            manager.m_boards[static_cast<std::size_t>(i)]
                .getAlphaZeroTensor(resultat.tenseurs[j]);

            manager.m_waiting_leaves.push_back(leaf);
            manager.m_waiting_game_indices.push_back(i);
            manager.m_waiting_moves_played.push_back(moves_played);
            manager.m_waiting_reservations.push_back(std::move(reservation));
            manager.m_is_waiting[i] = true;

            const std::size_t offset = manager.m_waiting_leaves.size() - 1;
            std::copy(resultat.tenseurs[j].begin(),
                      resultat.tenseurs[j].end(),
                      manager.m_batch_input.begin() + offset * 119 * 64);
        }

        manager.execute_gpu_batch();
        return resultat;
    }
};

namespace {

constexpr int GAME_COUNT = 8;
constexpr int INPUT_SIZE = 119 * 64;
constexpr int POLICY_SIZE = 4672;

using PriorsRacine = std::vector<std::pair<int, float>>;

PriorsRacine priors_de_la_racine(MCTSNode* root) {
    PriorsRacine priors;
    if (root == nullptr) return priors;
    for (const auto& enfant : root->children) {
        priors.emplace_back(enfant.first, enfant.second->prior);
    }
    std::sort(priors.begin(), priors.end());
    return priors;
}

bool priors_egaux(const PriorsRacine& gauche, const PriorsRacine& droite) {
    if (gauche.size() != droite.size()) return false;
    for (std::size_t i = 0; i < gauche.size(); ++i) {
        if (gauche[i].first != droite[i].first
            || std::fabs(gauche[i].second - droite[i].second) > 1e-7f) {
            return false;
        }
    }
    return true;
}

// Priors d'une racine de depart developpee sans aucun bruit : la reference a
// laquelle un coup rapide sans bruit doit rester identique.
PriorsRacine priors_reseau() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::preparer_position(manager, 0, "");
    return priors_de_la_racine(SelfPlayTestAccess::root(manager, 0));
}

void test_openmp_selfplay_core_with_shared_cache() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 3, 0);
    std::vector<Chessboard> boards(GAME_COUNT);
    std::vector<std::unique_ptr<MCTSNode>> roots;
    roots.reserve(GAME_COUNT);
    for (int game = 0; game < GAME_COUNT; ++game) {
        boards[game].setStartupPieces();
        roots.push_back(std::make_unique<MCTSNode>(0.0f));
        mcts.expand_node_single(roots.back().get(), boards[game]);
    }

    std::vector<MCTSNode*> leaves(GAME_COUNT);
    std::vector<int> moves_played(GAME_COUNT);
    std::vector<PathReservation> reservations(GAME_COUNT);
    std::vector<std::vector<float>> tensors(
        GAME_COUNT, std::vector<float>(INPUT_SIZE));
    std::vector<float> input;
    std::vector<float> policies;
    std::vector<float> values;

    for (int round = 0; round < 64; ++round) {
        std::fill(leaves.begin(), leaves.end(), nullptr);
        std::fill(moves_played.begin(), moves_played.end(), 0);

#pragma omp parallel for num_threads(GAME_COUNT) schedule(static)
        for (int game = 0; game < GAME_COUNT; ++game) {
            leaves[game] = mcts.advance_to_leaf(
                roots[game].get(), boards[game], 1.4f,
                moves_played[game], reservations[game]);
            if (leaves[game] != nullptr) {
                boards[game].getAlphaZeroTensor(tensors[game]);
            }
        }

        input.clear();
        std::vector<int> active;
        for (int game = 0; game < GAME_COUNT; ++game) {
            if (leaves[game] == nullptr) continue;
            active.push_back(game);
            input.insert(input.end(), tensors[game].begin(),
                         tensors[game].end());
        }
        if (!active.empty()) {
            evaluator.evaluate_batch(
                input, policies, values, static_cast<int>(active.size()));
            for (std::size_t index = 0; index < active.size(); ++index) {
                const int game = active[index];
                mcts.expand_and_backup(
                    leaves[game], boards[game],
                    policies.data() + index * POLICY_SIZE,
                    values[index], reservations[game]);
                for (int move = 0; move < moves_played[game]; ++move) {
                    boards[game].undoMove();
                }
            }
        }
    }

    for (const auto& root : roots) {
        const TreeReport report = mcts.inspect_tree(root.get());
        require_test(report.root_visits == 64,
                     "shared self-play root lost visits");
        require_test(report.en_vol == 0 && report.pending == 0,
                     "shared self-play root retained a reservation");
        require_test(report.violations == 0,
                     "shared self-play root is invalid");
    }
}

void test_selfplay_manager_with_controlled_evaluator() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    const std::vector<GameResult> games = manager.generate_games(2);

    require_test(games.size() == 2, "self-play returned the wrong game count");
    for (const GameResult& game : games) {
        require_test(game.move_count >= 0, "negative training move count");
        // Sans puzzle, aucun historique rejoue : la longueur ne peut pas
        // depasser le plafond de demi-coups joues.
        require_test(game.total_real_moves > 0
                         && game.total_real_moves
                                <= SelfPlayTestAccess::plafond_plies(),
                     "self-play game length is outside its bounds");
        require_test(game.flat_states.size()
                         == static_cast<std::size_t>(game.move_count)
                             * INPUT_SIZE,
                     "self-play state dimensions are inconsistent");
        require_test(game.flat_policies.size()
                         == static_cast<std::size_t>(game.move_count)
                             * POLICY_SIZE,
                     "self-play policy dimensions are inconsistent");
        require_test(std::all_of(
            game.flat_states.begin(), game.flat_states.end(),
            [](float value) { return std::isfinite(value); }),
            "self-play emitted a non-finite state");
        require_test(std::all_of(
            game.flat_policies.begin(), game.flat_policies.end(),
            [](float value) { return std::isfinite(value); }),
            "self-play emitted a non-finite policy");
        require_test(std::isfinite(game.final_outcome),
                     "self-play emitted a non-finite outcome");
        require_test(game.end_reason >= 0 && game.end_reason <= 5,
                     "end reason is out of range");
        if (game.end_reason == 0) {
            require_test(game.final_outcome == 1.0f
                             || game.final_outcome == -1.0f,
                         "checkmate without a decisive outcome");
        }
        else {
            require_test(game.final_outcome == 0.0f,
                         "non-checkmate game with a decisive outcome");
        }
    }
}

void test_selfplay_rejects_invalid_batch_and_cleans_up() {
    ControlledEvaluator evaluator;
    // Deux expansions de racine a la construction, puis le premier lot GPU.
    evaluator.corrupt_call = 3;
    evaluator.values_corruption =
        ControlledEvaluator::Corruption::NonFinite;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    bool failed = false;
    try {
        (void)manager.generate_games(2);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "self-play accepted an invalid batch output");

    // Les reservations et les plateaux en attente doivent avoir ete nettoyes :
    // le gestionnaire reste utilisable apres le rejet.
    evaluator.corrupt_call = 0;
    const std::vector<GameResult> games = manager.generate_games(2);
    require_test(games.size() == 2, "self-play did not recover");
}

void test_selfplay_roots_are_expanded_and_noised_exactly_once() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    // Le constructeur ne demarre plus de partie : le quota n'est connu que dans
    // generate_games. On demarre deux places pour controler les racines.
    SelfPlayTestAccess::demarrer_slot(manager, 0);
    SelfPlayTestAccess::demarrer_slot(manager, 1);

    // La deuxieme partie partage la position de depart de la premiere : sa
    // racine est servie par la table et doit tout de meme avoir ses enfants et
    // son bruit de Dirichlet.
    for (int game = 0; game < 2; ++game) {
        MCTSNode* root = SelfPlayTestAccess::root(manager, game);
        require_test(root != nullptr, "missing self-play root");
        require_test(root->state.load() == NodeState::Expanded,
                     "self-play root was left unexpanded");
        require_test(!root->children.empty(),
                     "self-play root has no children");
        require_test(SelfPlayTestAccess::pending_epsilon(manager, game) == 0.0f,
                     "self-play root kept a pending noise");
    }
}

void test_selfplay_generation_is_finite_and_counted() {
    struct Cas {
        int places;
        int quota;
    };
    const Cas cas[] = {{2, 3}, {4, 0}, {4, 1}, {4, 3}, {4, 4}, {4, 5},
                       {4, 7}};

    for (const Cas& c : cas) {
        ControlledEvaluator evaluator;
        SelfPlayManager manager(&evaluator, c.places, 2, 1, 0.5f, 8192);
        const std::vector<GameResult> games = manager.generate_games(c.quota);
        const SelfPlayStats stats = manager.get_stats();

        require_test(games.size() == static_cast<std::size_t>(c.quota),
                     "the generation did not return exactly N games");
        require_test(stats.games_started == static_cast<std::uint64_t>(c.quota),
                     "the number of starts does not match the quota");
        require_test(stats.games_completed == static_cast<std::uint64_t>(c.quota),
                     "the number of completions does not match the quota");
        require_test(stats.active_slots == 0,
                     "a place stayed active after the generation");
        require_test(stats.replayed_plies == 0,
                     "replayed plies were counted without puzzles");
        if (c.quota == 0) {
            require_test(stats.new_plies == 0, "N=0 played a move");
            require_test(evaluator.batch_sizes.empty(),
                         "N=0 called the evaluator");
        }
        else {
            require_test(stats.new_plies > 0, "no new ply was counted");
        }
    }
}

void test_successive_selfplay_generations_are_independent() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    const std::vector<GameResult> premier = manager.generate_games(2);
    const SelfPlayStats stats_premier = manager.get_stats();
    const std::vector<GameResult> second = manager.generate_games(3);
    const SelfPlayStats stats_second = manager.get_stats();

    require_test(premier.size() == 2, "the first generation is incomplete");
    require_test(second.size() == 3, "the second generation is incomplete");
    require_test(stats_premier.games_completed == 2,
                 "the first counters were changed by the second call");
    require_test(stats_second.games_started == 3
                     && stats_second.games_completed == 3,
                 "the second generation did not restart its counters");
}

void test_selfplay_batch_associates_each_game_with_its_own_tensor() {
    DiscriminatingEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 1, 1, 0.5f, 8192);

    // Deux positions distinctes : les tenseurs et les reponses different, un
    // decalage entre lignes du lot et parties devient visible.
    SelfPlayTestAccess::preparer_position(manager, 0, "e2e4");
    SelfPlayTestAccess::preparer_position(manager, 1, "d2d4");
    const std::string fen0 = SelfPlayTestAccess::fen(manager, 0);
    const std::string fen1 = SelfPlayTestAccess::fen(manager, 1);

    const SelfPlayTestAccess::ResultatLot lot =
        SelfPlayTestAccess::executer_lot_par_partie(manager, {0, 1});

    for (std::size_t j = 0; j < 2; ++j) {
        require_test(lot.feuilles[j] != nullptr,
                     "a game produced no network leaf");
        const auto& reponse = evaluator.reponse_pour(lot.tenseurs[j]);
        require_test(
            std::fabs(lot.feuilles[j]->network_value - reponse.value) < 1e-6f,
            "a game received another game's value");
        require_test(lot.feuilles[j]->visit_count == 1,
                     "the leaf did not receive exactly one backup");

        float somme = 0.0f;
        for (int idx : lot.coups_legaux[j]) {
            somme += reponse.policy[static_cast<std::size_t>(idx)];
        }
        require_test(somme > 0.0f, "the response has no legal mass");
        require_test(lot.feuilles[j]->children.size()
                         == lot.coups_legaux[j].size(),
                     "the leaf did not expose its legal children");
        for (const auto& child : lot.feuilles[j]->children) {
            const float attendu =
                reponse.policy[static_cast<std::size_t>(child.first)] / somme;
            require_test(std::fabs(child.second->prior - attendu) < 1e-5f,
                         "a game received another game's policy");
        }

        MCTSNode* racine =
            SelfPlayTestAccess::root(manager, static_cast<int>(j));
        int profondeur = 0;
        for (MCTSNode* n = lot.feuilles[j]; n->parent != nullptr;
             n = n->parent) {
            ++profondeur;
        }
        const float signe = (profondeur % 2 == 0) ? 1.0f : -1.0f;
        require_test(
            std::fabs(racine->total_value - signe * reponse.value) < 1e-6f,
            "the backup sign is wrong at the root");
    }

    require_test(SelfPlayTestAccess::fen(manager, 0) == fen0,
                 "the first board was not restored");
    require_test(SelfPlayTestAccess::fen(manager, 1) == fen1,
                 "the second board was not restored");
}

void test_one_active_slot_finishes_with_partial_batches() {
    // Une place seule ne peut jamais remplir un lot de deux : la fin de
    // generation doit avancer par lots partiels, sans attendre un lot plein et
    // sans toucher la place inactive restee sans racine.
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    const std::vector<GameResult> games = manager.generate_games(1);
    const SelfPlayStats stats = manager.get_stats();

    require_test(games.size() == 1, "the single game was not collected");
    require_test(stats.games_started == 1 && stats.games_completed == 1,
                 "the single active slot did not start and finish once");
    require_test(stats.active_slots == 0, "a place stayed active");
    require_test(!evaluator.batch_sizes.empty(),
                 "the single game never reached the evaluator");
    for (int taille : evaluator.batch_sizes) {
        require_test(taille <= 1, "the drain waited for a fuller batch");
    }
    require_test(SelfPlayTestAccess::root(manager, 1) == nullptr,
                 "the inactive slot received a root");
}

void test_imposed_ends_start_and_collect_each_game_once() {
    // Une place finit tot, l'autre tard, et les fins sont imposees a des
    // moments connus : aucune attente d'horloge, aucun terminal aleatoire.
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::forcer_fin_apres(manager, 0, 2);
    SelfPlayTestAccess::forcer_fin_apres(manager, 1, 8);

    const std::vector<GameResult> games = manager.generate_games(3);
    const SelfPlayStats stats = manager.get_stats();

    require_test(games.size() == 3,
                 "the generation did not collect exactly three games");
    require_test(stats.games_started == 3 && stats.games_completed == 3,
                 "the starts and completions do not match the quota");
    require_test(stats.active_slots == 0,
                 "a place stayed active after the generation");

    // Identites : la place 0 a joue deux parties de 2 plies, la place 1 une
    // partie de 8. Aucune partie fantome, aucune collecte en double.
    std::vector<int> longueurs;
    for (const GameResult& game : games) {
        longueurs.push_back(game.total_real_moves);
    }
    std::sort(longueurs.begin(), longueurs.end());
    require_test(longueurs == std::vector<int>({2, 2, 8}),
                 "the collected games do not match the imposed ends");
}

std::string ecrire_fixture_puzzle() {
    const std::string chemin = "puzzles_test_fixture.txt";
    std::ofstream fichier(chemin);
    fichier << "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
               "|e2e4 e7e5|e7e5|1500|fork\n";
    return chemin;
}

bool chercher_graine_injectee(SelfPlayManager& manager) {
    // L'injection est tiree au hasard : on cherche une graine qui la
    // declenche, sans attendre ni dependre du temps.
    for (std::uint32_t graine = 1; graine <= 2000; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        SelfPlayTestAccess::reset_game(manager, 0);
        if (SelfPlayTestAccess::tactical_boost(manager, 0)) {
            return true;
        }
    }
    return false;
}

// Allers-retours de cavaliers, legaux depuis le depart comme apres 1.e4 e5 :
// ils allongent une partie sans la terminer, puisque movePieceUCI ne teste pas
// la fin de partie.
// debut est le rang du premier demi-coup dans le cycle, pour reprendre la ou
// un appel precedent s'est arrete.
void jouer_allers_retours(SelfPlayManager& manager, int game_idx, int debut,
                          int plies) {
    static const char* const cycle[] = {"g1f3", "g8f6", "f3g1", "f6g8"};
    for (int k = debut; k < debut + plies; ++k) {
        SelfPlayTestAccess::jouer_uci(manager, game_idx, cycle[k % 4]);
    }
}

void test_length_cap_is_four_hundred_plies_for_a_normal_game() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);
    const int plafond = SelfPlayTestAccess::plafond_plies();
    require_test(plafond == 400, "the self-play length cap is not 400 plies");

    SelfPlayTestAccess::reset_game(manager, 0);
    jouer_allers_retours(manager, 0, 0, plafond - 1);
    require_test(!SelfPlayTestAccess::longueur_max_atteinte(manager, 0),
                 "a normal game was capped before the limit");
    jouer_allers_retours(manager, 0, plafond - 1, 1);
    require_test(SelfPlayTestAccess::longueur_max_atteinte(manager, 0),
                 "a normal game was not capped at the limit");
}

void test_puzzle_replay_does_not_count_toward_the_length_cap() {
    const std::string fixture = ecrire_fixture_puzzle();  // rejoue e2e4 e7e5
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());
    require_test(chercher_graine_injectee(manager),
                 "no seed injected the puzzle");
    const int plafond = SelfPlayTestAccess::plafond_plies();

    jouer_allers_retours(manager, 0, 0, plafond - 1);
    require_test(!SelfPlayTestAccess::longueur_max_atteinte(manager, 0),
                 "the replayed puzzle history counted toward the cap");
    jouer_allers_retours(manager, 0, plafond - 1, 1);
    require_test(SelfPlayTestAccess::longueur_max_atteinte(manager, 0),
                 "a puzzle game was not capped after the limit of played plies");
    require_test(SelfPlayTestAccess::historique(manager, 0) == plafond + 2,
                 "the board history no longer holds the replayed plies");
}

std::string ecrire_fixture_depart() {
    const std::string chemin = "positions_depart_fixture.txt";
    std::ofstream fichier(chemin);
    fichier << "# finale de pions\n"
               "8/5pk1/6p1/3pP3/3P1P2/6K1/8/8 w - - 0 40\n\n";
    return chemin;
}

void test_start_positions_replace_the_initial_position_and_puzzles() {
    const std::string depart = ecrire_fixture_depart();
    const std::string puzzles = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, puzzles, depart);
    std::remove(depart.c_str());
    std::remove(puzzles.c_str());

    // Aucune graine ne doit injecter de puzzle : les positions de depart
    // remplacent la position initiale comme les puzzles.
    for (std::uint32_t graine = 1; graine <= 200; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        SelfPlayTestAccess::reset_game(manager, 0);
        require_test(!SelfPlayTestAccess::tactical_boost(manager, 0),
                     "a start position game received the puzzle boost");
        const std::string fen = SelfPlayTestAccess::fen(manager, 0);
        require_test(fen.rfind("8/5pk1/6p1/3pP3/3P1P2/6K1/8/8 w", 0) == 0,
                     "a game did not start from the start position file");
    }
    require_test(SelfPlayTestAccess::historique(manager, 0) == 0,
                 "a start position game has a replayed history");
}

void test_an_unreadable_start_position_file_fails_loudly() {
    ControlledEvaluator evaluator;
    bool leve = false;
    try {
        SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192,
                                "absent.txt", "positions_absentes.txt");
    }
    catch (const std::runtime_error&) {
        leve = true;
    }
    require_test(leve, "a missing start position file was silently ignored");
}

void test_puzzle_first_move_boost_then_normal_move() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    require_test(chercher_graine_injectee(manager),
                 "no seed injected the puzzle");
    require_test(SelfPlayTestAccess::sims_target(manager, 0) == 4000,
                 "the puzzle first move lost its tactical budget");
    require_test(SelfPlayTestAccess::is_slow_move(manager, 0),
                 "the puzzle first move is not a slow move");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the tactical noise was not consumed");
    require_test(SelfPlayTestAccess::root(manager, 0) != nullptr,
                 "the puzzle root is missing");

    // Le coup suivant retrouve un budget normal et abandonne l'epsilon
    // tactique : le bruit normal est soit deja applique, soit en attente.
    SelfPlayTestAccess::play_best_move(manager, 0);
    require_test(!SelfPlayTestAccess::tactical_boost(manager, 0),
                 "the tactical boost was not consumed");
    const int cible = SelfPlayTestAccess::sims_target(manager, 0);
    require_test(cible == 1 || cible == 2,
                 "the next move did not return to a normal budget");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) <= 0.12f,
                 "the tactical epsilon survived the first move");
}

void test_puzzle_first_move_served_by_the_table_keeps_the_boost() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    // Prechauffe la table avec la position du puzzle.
    Chessboard position;
    position.loadFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    require_test(position.movePieceUCI("e2e4"), "e2e4 failed");
    require_test(position.movePieceUCI("e7e5"), "e7e5 failed");
    const std::vector<int> legal = position.getLegalMoveIndices();
    std::vector<float> policy(POLICY_SIZE, 0.0f);
    for (int idx : legal) policy[static_cast<std::size_t>(idx)] = 1.0f;
    MCTSTestAccess::store(SelfPlayTestAccess::mcts(manager), position, legal,
                          policy, 0.1f);

    require_test(chercher_graine_injectee(manager),
                 "no seed injected the puzzle");
    MCTSNode* root = SelfPlayTestAccess::root(manager, 0);
    require_test(root != nullptr, "the puzzle root is missing");
    require_test(root->state.load() == NodeState::Expanded,
                 "a TT hit left the puzzle root undeveloped");
    require_test(!root->children.empty(),
                 "a TT hit lost the puzzle root children");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the tactical noise was not consumed on a TT hit");
    require_test(SelfPlayTestAccess::sims_target(manager, 0) == 4000,
                 "a TT hit lost the tactical budget");
}

void test_reused_root_has_children_and_consumes_its_noise() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    SelfPlayTestAccess::preparer_position(manager, 0, "");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the prepared root kept a pending noise");

    // Une vague developpe un enfant, puis le meilleur coup le promeut en
    // racine. La nouvelle racine doit deja porter ses enfants et avoir
    // consomme son bruit, sans dependre d'un passage GPU ulterieur : les deux
    // sites qui posent le bruit (reset_game et play_best_move) developpent la
    // racine avant de l'appliquer, donc la descente ne materialise jamais une
    // racine sans enfants depuis la table.
    SelfPlayTestAccess::executer_lot_par_partie(manager, {0});
    SelfPlayTestAccess::play_best_move(manager, 0);

    MCTSNode* racine = SelfPlayTestAccess::root(manager, 0);
    require_test(racine != nullptr, "the reused root is missing");
    require_test(!racine->children.empty(),
                 "the reused root has no children");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the reused root kept its noise pending");
}

void test_epsilon_for_move_table() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);

    // Mode historique : les deux types de coups recoivent le bruit normal.
    SelfPlayTestAccess::set_is_slow_move(manager, 0, false);
    SelfPlayTestAccess::set_sims_target(manager, 0, 100);
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.12f,
                 "a fast move lost its historical noise");
    SelfPlayTestAccess::set_is_slow_move(manager, 0, true);
    SelfPlayTestAccess::set_sims_target(manager, 0, 700);
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.12f,
                 "a slow move changed its historical noise");

    // Nouvelle campagne : le bruit rapide disparait et la classification prime
    // sur le nombre de simulations.
    manager.set_fast_dirichlet_noise(false);
    SelfPlayTestAccess::set_is_slow_move(manager, 0, false);
    SelfPlayTestAccess::set_sims_target(manager, 0, 700);
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.0f,
                 "a fast move with a slow budget kept its noise");
    SelfPlayTestAccess::set_is_slow_move(manager, 0, true);
    SelfPlayTestAccess::set_sims_target(manager, 0, 100);
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.12f,
                 "a slow move with a fast budget lost its noise");

    // Le renfort du premier coup de puzzle reste prioritaire dans les deux
    // modes, meme classe rapide.
    SelfPlayTestAccess::set_tactical_boost(manager, 0, true);
    SelfPlayTestAccess::set_is_slow_move(manager, 0, false);
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.30f,
                 "the puzzle boost lost priority with fast noise off");
    manager.set_fast_dirichlet_noise(true);
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.30f,
                 "the puzzle boost lost priority with fast noise on");
}

void test_fast_move_without_noise_keeps_priors_and_rng() {
    const PriorsRacine reference = priors_reseau();
    require_test(!reference.empty(), "the reference root has no prior");

    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);
    manager.set_fast_dirichlet_noise(false);
    bool trouve = false;
    for (std::uint32_t graine = 1; graine <= 2000 && !trouve; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        MCTSTestAccess::seed_noise(SelfPlayTestAccess::mcts(manager), 123);
        const std::mt19937 avant = MCTSTestAccess::noise_rng_state(
            SelfPlayTestAccess::mcts(manager));
        SelfPlayTestAccess::reset_game(manager, 0);
        const std::mt19937 apres = MCTSTestAccess::noise_rng_state(
            SelfPlayTestAccess::mcts(manager));
        if (SelfPlayTestAccess::is_slow_move(manager, 0)) continue;
        trouve = true;
        require_test(avant == apres, "a fast move consumed the noise RNG");
        require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                     "a fast move kept a pending noise");
        require_test(
            priors_egaux(priors_de_la_racine(
                             SelfPlayTestAccess::root(manager, 0)),
                         reference),
            "a fast move changed the network priors");
    }
    require_test(trouve, "no seed produced a fast move");
}

void test_slow_move_applies_noise_exactly_once() {
    const PriorsRacine reference = priors_reseau();

    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);
    bool trouve = false;
    for (std::uint32_t graine = 1; graine <= 2000 && !trouve; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        MCTSTestAccess::seed_noise(SelfPlayTestAccess::mcts(manager), 42);
        SelfPlayTestAccess::reset_game(manager, 0);
        if (!SelfPlayTestAccess::is_slow_move(manager, 0)) continue;
        trouve = true;
        require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                     "the slow noise was not applied to the developed root");
        const PriorsRacine bruitees = priors_de_la_racine(
            SelfPlayTestAccess::root(manager, 0));
        require_test(!priors_egaux(reference, bruitees),
                     "the slow move did not perturb the priors");
        SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
        require_test(priors_egaux(bruitees, priors_de_la_racine(
                                                SelfPlayTestAccess::root(
                                                    manager, 0))),
                     "the slow noise was applied twice");
    }
    require_test(trouve, "no seed produced a slow move");
}

void test_deferred_noise_once_and_zero_inert() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);
    manager.set_fast_dirichlet_noise(false);

    // Racine sans enfants : le bruit positif reste en attente.
    SelfPlayTestAccess::racine_vide(manager, 0);
    SelfPlayTestAccess::set_pending_epsilon(manager, 0, 0.12f);
    SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
    SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.12f,
                 "a childless root consumed the pending noise");

    SelfPlayTestAccess::developper_racine(manager, 0);
    const PriorsRacine reseau = priors_de_la_racine(
        SelfPlayTestAccess::root(manager, 0));
    require_test(!reseau.empty(), "the developed root has no prior");
    SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the first application did not consume the noise");
    const PriorsRacine bruitees = priors_de_la_racine(
        SelfPlayTestAccess::root(manager, 0));
    require_test(!priors_egaux(reseau, bruitees),
                 "the positive noise did not change the priors");
    SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
    require_test(priors_egaux(bruitees, priors_de_la_racine(
                                               SelfPlayTestAccess::root(
                                                   manager, 0))),
                 "the positive noise was applied twice");

    // Bruit nul : les deux applications restent inertes, sans tirage.
    SelfPlayTestAccess::racine_vide(manager, 0);
    SelfPlayTestAccess::set_is_slow_move(manager, 0, false);
    SelfPlayTestAccess::set_tactical_boost(manager, 0, false);
    const float epsilon = SelfPlayTestAccess::epsilon_for_move(manager, 0);
    require_test(epsilon == 0.0f, "a fast move did not return a zero epsilon");
    SelfPlayTestAccess::set_pending_epsilon(manager, 0, epsilon);
    SelfPlayTestAccess::developper_racine(manager, 0);
    const PriorsRacine avant = priors_de_la_racine(
        SelfPlayTestAccess::root(manager, 0));
    const std::mt19937 rng_avant =
        MCTSTestAccess::noise_rng_state(SelfPlayTestAccess::mcts(manager));
    SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
    SelfPlayTestAccess::appliquer_bruit_en_attente(manager, 0);
    const std::mt19937 rng_apres =
        MCTSTestAccess::noise_rng_state(SelfPlayTestAccess::mcts(manager));
    require_test(rng_avant == rng_apres,
                 "a zero noise consumed the noise RNG");
    require_test(priors_egaux(avant, priors_de_la_racine(
                                            SelfPlayTestAccess::root(
                                                manager, 0))),
                 "a zero noise changed the priors");
}

void test_fast_reused_root_preserves_visits_and_values() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192);
    manager.set_fast_dirichlet_noise(false);
    SelfPlayTestAccess::preparer_position(manager, 0, "");
    SelfPlayTestAccess::executer_lot_par_partie(manager, {0});

    MCTSNode* racine = SelfPlayTestAccess::root(manager, 0);
    MCTSNode* promu = nullptr;
    for (const auto& enfant : racine->children) {
        if (enfant.second->visit_count > 0) {
            promu = enfant.second.get();
            break;
        }
    }
    require_test(promu != nullptr, "no child received a visit");
    const int visites = promu->visit_count;
    const float valeur = promu->total_value;

    SelfPlayTestAccess::play_best_move(manager, 0);
    require_test(SelfPlayTestAccess::root(manager, 0) == promu,
                 "the visited child was not promoted to the root");
    require_test(visites > 0 && promu->visit_count == visites
                     && promu->total_value == valeur,
                 "promotion changed the child visits or value");
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) < 0.30f,
                 "a normal move inherited the tactical epsilon");
}

void test_puzzle_boost_survives_fast_noise_disabled() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());
    manager.set_fast_dirichlet_noise(false);

    require_test(chercher_graine_injectee(manager),
                 "no seed injected the puzzle");
    require_test(SelfPlayTestAccess::tactical_boost(manager, 0),
                 "the injected puzzle lost its boost");
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) == 0.30f,
                 "the puzzle boost is not 0.30 with fast noise off");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the puzzle noise was not applied to the developed root");

    SelfPlayTestAccess::play_best_move(manager, 0);
    require_test(!SelfPlayTestAccess::tactical_boost(manager, 0),
                 "the boost survived the first played move");
    require_test(SelfPlayTestAccess::epsilon_for_move(manager, 0) < 0.30f,
                 "0.30 survived the first played move");
}

void test_slow_puzzle_game_is_not_lost_behind_a_fast_one() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    // Une place joue un puzzle, premier coup a 4000 simulations, et doit etre
    // recuperee malgre la fin rapide de l'autre. La graine est cherchee sur les
    // demarrages reellement consommes par generate_games : sans historique
    // rejoue, les longueurs {2,4} viendraient des seules fins forcees et le
    // test ne prouverait rien.
    SelfPlayTestAccess::forcer_fin_apres(manager, 0, 4);
    SelfPlayTestAccess::forcer_fin_apres(manager, 1, 2);

    bool trouve = false;
    std::vector<GameResult> games;
    SelfPlayStats stats;
    for (std::uint32_t graine = 1; graine <= 400 && !trouve; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        games = manager.generate_games(2);
        stats = manager.get_stats();

        std::vector<int> longueurs;
        for (const GameResult& game : games) {
            longueurs.push_back(game.total_real_moves);
        }
        std::sort(longueurs.begin(), longueurs.end());
        trouve = stats.replayed_plies == 2
            && longueurs == std::vector<int>({2, 4});
    }
    require_test(trouve,
                 "no seed gave a slow puzzle game behind a fast one");

    require_test(games.size() == 2, "the slow game was not collected");
    require_test(stats.games_started == 2 && stats.games_completed == 2,
                 "the starts and completions do not match the quota");
    require_test(stats.active_slots == 0, "a place stayed active");

    // La partie de 4 plies est celle du puzzle : deux plies rejoues puis deux
    // joues, dont le premier coup tactique enregistre comme slow move.
    const GameResult* puzzle = nullptr;
    for (const GameResult& game : games) {
        if (game.total_real_moves == 4) puzzle = &game;
    }
    require_test(puzzle != nullptr && puzzle->move_count >= 1,
                 "the tactical first move was not recorded");
}

void test_game_conclusion_signs_and_reasons() {
    Chessboard board;

    board.loadFEN("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    GameConclusion c = conclure_partie(board, false);
    require_test(c.end_reason == 0 && c.final_outcome == 1.0f,
                 "a mated black side should be a win for White");

    board.loadFEN("8/8/8/8/8/5k2/6q1/7K w - - 0 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 0 && c.final_outcome == -1.0f,
                 "a mated white side should be a loss for White");

    board.loadFEN("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 1 && c.final_outcome == 0.0f,
                 "stalemate should be a draw");

    board.loadFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    c = conclure_partie(board, true);
    require_test(c.end_reason == 5 && c.final_outcome == 0.0f,
                 "the maximum length should be a draw");

    board.setStartupPieces();
    const char* moves[] = {"g1f3", "g8f6", "f3g1", "f6g8",
                           "g1f3", "g8f6", "f3g1", "f6g8"};
    for (const char* uci : moves) {
        require_test(board.movePieceUCI(uci), "repetition move failed");
    }
    c = conclure_partie(board, false);
    require_test(c.end_reason == 2 && c.final_outcome == 0.0f,
                 "repetition should be a draw");

    board.loadFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 3 && c.final_outcome == 0.0f,
                 "the fifty-move rule should be a draw");

    board.loadFEN("8/8/8/8/8/8/8/K6k w - - 0 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 4 && c.final_outcome == 0.0f,
                 "insufficient material should be a draw");
}

std::vector<GameResult> generer_avec_reglages(int virtual_loss,
                                              std::uint32_t graine) {
    DiscriminatingEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::seed_rng(manager, graine);
    MCTSTestAccess::seed_noise(SelfPlayTestAccess::mcts(manager), graine);
    SelfPlayTestAccess::mcts(manager).set_tuning(
        SearchTuning{virtual_loss, 0.30f, 4});
    return manager.generate_games(2);
}

void test_virtual_loss_is_inert_in_self_play() {
    // Chaque arbre n'est descendu qu'une fois par vague : n_in_flight n'a pas
    // de lecteur entre la reservation et sa liberation. Deux generations de
    // meme graine doivent donc etre identiques, quelle que soit l'amplitude.
    const std::vector<GameResult> reference = generer_avec_reglages(1, 12345);
    const std::vector<GameResult> candidat = generer_avec_reglages(2, 12345);

    require_test(reference.size() == candidat.size(),
                 "the two generations do not have the same game count");
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require_test(reference[i].flat_states == candidat[i].flat_states,
                     "the amplitude changed the self-play states");
        require_test(reference[i].flat_policies == candidat[i].flat_policies,
                     "the amplitude changed the self-play policies");
        require_test(reference[i].final_outcome == candidat[i].final_outcome
                         && reference[i].end_reason == candidat[i].end_reason,
                     "the amplitude changed the self-play outcome");
    }
}

void test_descent_work_counts_levels_and_children_exactly() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192);
    Chessboard board;
    board.setStartupPieces();
    MCTSNode root(0.0f);
    SearchWorkCounters work;
    PathReservation reservation;
    int depth = -1;
    require_test(mcts.advance_to_leaf(
                     &root, board, 1.4f, depth, reservation, &work) == &root,
                 "an empty root did not yield a leaf");
    reservation.release();
    require_test(work.descents == 1 && work.depth_sum == 0
                     && work.depth_max == 0 && work.children_scored == 0,
                 "a depth-zero descent was counted incorrectly");

    const auto root_moves = board.getLegalMoveIndices();
    auto branch = std::make_unique<MCTSNode>(1.0f, root_moves[0], &root);
    MCTSNode* first = branch.get();
    root.children.emplace_back(root_moves[0], std::move(branch));
    root.children.emplace_back(root_moves[1],
        std::make_unique<MCTSNode>(0.0f, root_moves[1], &root));
    root.visit_count = 1;
    root.state.store(NodeState::Expanded);
    Chessboard next = board;
    require_test(mcts.apply_move_by_index(next, root_moves[0]),
                 "the test move was refused");
    const auto next_moves = next.getLegalMoveIndices();
    first->children.emplace_back(next_moves[0],
        std::make_unique<MCTSNode>(1.0f, next_moves[0], first));
    first->children.emplace_back(next_moves[1],
        std::make_unique<MCTSNode>(0.0f, next_moves[1], first));
    first->visit_count = 1;
    first->state.store(NodeState::Expanded);
    require_test(mcts.advance_to_leaf(
                     &root, board, 1.4f, depth, reservation, &work)
                     == first->children[0].second.get(),
                 "the controlled path changed");
    require_test(depth == 2 && work.descents == 2 && work.depth_sum == 2
                     && work.depth_max == 2 && work.children_scored == 4,
                 "the descent did not count both levels and all UCB children");
    reservation.release();
    board.undoMove();
    board.undoMove();
}

void test_diagnostics_are_inert_and_match_the_evaluator() {
    // Meme graine et meme bruit : activer le diagnostic ne doit changer ni le
    // deroulement ni les parties produites.
    const std::uint32_t graine = 424242;

    ControlledEvaluator evaluator_reference;
    std::vector<GameResult> reference;
    {
        SelfPlayManager manager(&evaluator_reference, 2, 2, 1, 0.5f, 8192);
        SelfPlayTestAccess::seed_rng(manager, graine);
        MCTSTestAccess::seed_noise(SelfPlayTestAccess::mcts(manager), graine);
        reference = manager.generate_games(2);
        const SelfPlayTiming timing = manager.get_timing();
        require_test(!timing.enabled,
                     "the disabled diagnostic is marked enabled");
        require_test(timing.batch_calls == 0 && timing.batch_rows == 0
                         && timing.generation_wall_ns == 0,
                     "the disabled diagnostic filled its report");
    }

    ControlledEvaluator evaluator_diagnostic;
    std::vector<GameResult> candidate;
    SelfPlayTiming timing;
    {
        SelfPlayManager manager(&evaluator_diagnostic, 2, 2, 1, 0.5f, 8192);
        SelfPlayTestAccess::seed_rng(manager, graine);
        MCTSTestAccess::seed_noise(SelfPlayTestAccess::mcts(manager), graine);
        manager.set_diagnostics_mode(1);
        candidate = manager.generate_games(2);
        timing = manager.get_timing();
    }

    require_test(timing.search_descents == timing.completed_sims,
                 "a self-play descent was lost or counted twice");
    require_test(timing.search_depth_sum > 0 && timing.search_depth_max > 0
                     && timing.search_children_scored >= timing.search_depth_sum,
                 "the self-play work counters are inconsistent");
    require_test(timing.enabled, "the diagnostic did not turn on");
    require_test(timing.mode == 1, "the diagnostic reported the wrong mode");
    require_test(reference.size() == candidate.size(),
                 "the diagnostic changed the game count");
    std::uint64_t exemples_sauves = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require_test(reference[i].flat_states == candidate[i].flat_states,
                     "the diagnostic changed the self-play states");
        require_test(reference[i].flat_policies == candidate[i].flat_policies,
                     "the diagnostic changed the self-play policies");
        require_test(reference[i].final_outcome == candidate[i].final_outcome
                         && reference[i].end_reason == candidate[i].end_reason,
                     "the diagnostic changed the self-play outcome");
        exemples_sauves += static_cast<std::uint64_t>(
            candidate[i].move_count);
    }

    // Les compteurs reseau doivent coller aux appels reellement recus par
    // l'evaluateur, expansions de racine comprises.
    const std::uint64_t appels_physiques =
        static_cast<std::uint64_t>(evaluator_diagnostic.batch_sizes.size());
    require_test(timing.unit_network_calls + timing.batch_calls
                     == appels_physiques,
                 "the network call counters do not match the evaluator");
    std::uint64_t lignes_evaluees = 0;
    for (const int taille : evaluator_diagnostic.batch_sizes) {
        lignes_evaluees += static_cast<std::uint64_t>(taille);
    }
    require_test(timing.unit_network_rows + timing.batch_rows
                     == lignes_evaluees,
                 "the evaluated row counters do not match the evaluator");
    require_test(timing.unit_network_calls == timing.unit_network_rows,
                 "a root expansion evaluated more than one line");

    std::uint64_t histogramme = 0;
    for (const std::uint64_t compte : timing.batch_histogram) {
        histogramme += compte;
    }
    require_test(histogramme == timing.batch_calls,
                 "the batch histogram does not sum to the physical calls");
    require_test(timing.leaf_requests == timing.batch_rows,
                 "a collected request was never launched");
    require_test(timing.batch_calls > 0 && timing.batch_rows > 0,
                 "the diagnostic saw no network batch");
    require_test(timing.completed_sims
                     == timing.batch_rows + timing.no_network_sims,
                 "the completed simulation counter is inconsistent");
    require_test(timing.terminal_sims <= timing.no_network_sims,
                 "terminal simulations exceed the no-network simulations");
    require_test(timing.slow_examples_saved == exemples_sauves,
                 "the saved example counter does not match the results");
    require_test(timing.root_expansions > 0,
                 "the root expansion counter stayed empty");

    std::uint64_t phases_ns = 0;
    for (const std::uint64_t phase_ns : timing.phase_wall_ns) {
        phases_ns += phase_ns;
    }
    require_test(timing.generation_wall_ns >= phases_ns,
                 "the phases exceed the generation wall time");
    require_test(timing.generation_other_wall_ns
                     == static_cast<std::int64_t>(
                            timing.generation_wall_ns)
                        - static_cast<std::int64_t>(phases_ns),
                 "the residue does not close the phase budget");
    require_test(timing.deferred_turns <= timing.loop_turns,
                 "there are more deferred turns than turns");
    require_test(timing.deferred_wall_ns <= timing.generation_wall_ns,
                 "the deferred turns exceed the generation");
}

void test_diagnostics_mode_two_reports_worker_times() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 4, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::seed_rng(manager, 7);
    manager.set_diagnostics_mode(2);
    const std::vector<GameResult> games = manager.generate_games(4);
    const SelfPlayTiming timing = manager.get_timing();

    require_test(games.size() == 4, "mode 2 changed the game count");
    require_test(timing.mode == 2, "mode 2 was not reported");
    require_test(timing.worker_count == 4, "the wrong worker count was used");
    require_test(timing.worker_busy_max_ns > 0,
                 "no worker work was measured");
    require_test(timing.worker_busy_sum_ns >= timing.worker_busy_max_ns,
                 "the worker sum is smaller than its maximum");
    require_test(timing.worker_busy_sum_ns
                     <= timing.generation_wall_ns * timing.worker_count,
                 "the worker work exceeds the available wall time");
}

void test_diagnostics_are_reset_between_generations() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::seed_rng(manager, 99);
    manager.set_diagnostics_mode(1);

    manager.generate_games(2);
    const EvaluatorTotals before = evaluator.diagnostic_totals();
    manager.generate_games(2);
    const EvaluatorTotals after = evaluator.diagnostic_totals();
    const SelfPlayTiming second = manager.get_timing();

    require_test(second.search_descents == second.completed_sims,
                 "the descent counters were not reset");
    require_test(second.batch_calls + second.unit_network_calls
                     == after.run_calls - before.run_calls,
                 "the second generation accumulated the first counters");
    require_test(second.batch_rows + second.unit_network_rows
                     == after.evaluated_rows - before.evaluated_rows,
                 "the second generation accumulated the first rows");

    manager.set_diagnostics_mode(0);
    manager.generate_games(2);
    const SelfPlayTiming disabled = manager.get_timing();
    require_test(disabled.search_descents == 0 && disabled.search_depth_sum == 0
                     && disabled.search_depth_max == 0
                     && disabled.search_children_scored == 0,
                 "the disabled diagnostic recorded search work");
    require_test(!disabled.enabled && disabled.batch_calls == 0
                     && disabled.generation_wall_ns == 0,
                 "turning the diagnostic off left a stale report");
}

void test_diagnostics_measure_the_final_drain() {
    // Deux places pour six parties : la vidange commence au sixieme depart et
    // ne couvre qu'une partie de la generation.
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::seed_rng(manager, 7);
    manager.set_diagnostics_mode(1);
    manager.generate_games(6);
    const SelfPlayTiming partielle = manager.get_timing();
    require_test(partielle.drain_wall_ns > 0
                     && partielle.drain_wall_ns < partielle.generation_wall_ns,
                 "the drain does not cover only the end of the generation");
    require_test(partielle.drain_batch_calls <= partielle.batch_calls
                     && partielle.drain_batch_rows <= partielle.batch_rows,
                 "the drain counted more batches than the generation");

    // Autant de places que de parties : tout est vidange des le depart.
    manager.generate_games(2);
    const SelfPlayTiming totale = manager.get_timing();
    require_test(totale.drain_batch_calls == totale.batch_calls
                     && totale.drain_batch_rows == totale.batch_rows,
                 "a generation without refill is not entirely a drain");

    manager.set_diagnostics_mode(0);
    manager.generate_games(2);
    require_test(manager.get_timing().drain_wall_ns == 0,
                 "a disabled diagnostic measured a drain");
}

}  // namespace

int main() {
    try {
        test_openmp_selfplay_core_with_shared_cache();
        test_selfplay_manager_with_controlled_evaluator();
        test_selfplay_rejects_invalid_batch_and_cleans_up();
        test_selfplay_roots_are_expanded_and_noised_exactly_once();
        test_selfplay_generation_is_finite_and_counted();
        test_successive_selfplay_generations_are_independent();
        test_selfplay_batch_associates_each_game_with_its_own_tensor();
        test_one_active_slot_finishes_with_partial_batches();
        test_imposed_ends_start_and_collect_each_game_once();
        test_puzzle_first_move_boost_then_normal_move();
        test_puzzle_first_move_served_by_the_table_keeps_the_boost();
        test_length_cap_is_four_hundred_plies_for_a_normal_game();
        test_puzzle_replay_does_not_count_toward_the_length_cap();
        test_start_positions_replace_the_initial_position_and_puzzles();
        test_an_unreadable_start_position_file_fails_loudly();
        test_reused_root_has_children_and_consumes_its_noise();
        test_epsilon_for_move_table();
        test_fast_move_without_noise_keeps_priors_and_rng();
        test_slow_move_applies_noise_exactly_once();
        test_deferred_noise_once_and_zero_inert();
        test_fast_reused_root_preserves_visits_and_values();
        test_puzzle_boost_survives_fast_noise_disabled();
        test_slow_puzzle_game_is_not_lost_behind_a_fast_one();
        test_game_conclusion_signs_and_reasons();
        test_virtual_loss_is_inert_in_self_play();
        test_descent_work_counts_levels_and_children_exactly();
        test_diagnostics_are_inert_and_match_the_evaluator();
        test_diagnostics_mode_two_reports_worker_times();
        test_diagnostics_are_reset_between_generations();
        test_diagnostics_measure_the_final_drain();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
