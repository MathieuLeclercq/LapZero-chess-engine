#include "controlled_evaluator.hpp"
#include "mcts_test_access.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

class Rendezvous {
public:
    explicit Rendezvous(int participants) : m_remaining(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        --m_remaining;
        if (m_remaining == 0) {
            m_cv.notify_all();
            return;
        }
        m_cv.wait(lock, [&]() { return m_remaining == 0; });
    }

private:
    int m_remaining;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};

class PublicationGate {
public:
    void block_owner() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_blocked = true;
        m_cv.notify_all();
        m_cv.wait(lock, [&]() { return m_released; });
    }

    void wait_until_blocked() {
        std::unique_lock<std::mutex> lock(m_mutex);
        require_test(m_cv.wait_for(lock, 2s, [&]() { return m_blocked; }),
                     "TT owner did not reach publication gate");
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_released = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_blocked = false;
    bool m_released = false;
};

Chessboard startup_board() {
    Chessboard board;
    board.setStartupPieces();
    return board;
}

std::vector<float> uniform_policy() {
    return std::vector<float>(4672, 1.0f / 4672.0f);
}

std::unique_ptr<MCTSNode> expanded_root(
    MCTS& mcts, Chessboard& board, int visits);

std::vector<int> path_to(MCTSNode* root, MCTSNode* leaf) {
    std::vector<int> path;
    for (MCTSNode* node = leaf; node != nullptr && node != root;
         node = node->parent) {
        path.push_back(node->move_idx);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

void require_same_tensor(const std::vector<float>& left,
                         const std::vector<float>& right) {
    require_test(left.size() == right.size(), "tensor sizes differ");
    require_test(std::equal(left.begin(), left.end(), right.begin()),
                 "captured tensor does not match replayed path");
}

void test_only_one_collector_claims_a_leaf() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    MCTSNode root(0.0f);
    Chessboard board = startup_board();
    std::atomic<bool> cancelled{false};
    Rendezvous rendezvous(2);
    WaveTestHooks hooks;
    hooks.before_claim = [&](MCTSNode* node) {
        require_test(node == &root, "collectors reached different leaves");
        rendezvous.wait();
    };

    {
        std::array<WorkerContext, 2> contexts;
        contexts[0].board = board;
        contexts[1].board = board;
        std::array<LeafWork, 2> work;
        std::array<std::exception_ptr, 2> errors;
        std::array<std::thread, 2> threads;

        for (std::size_t i = 0; i < threads.size(); ++i) {
            threads[i] = std::thread([&, i]() {
                try {
                    work[i] = MCTSTestAccess::collect_leaf(
                        mcts, &root, contexts[i], 1.4f, cancelled, &hooks);
                }
                catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) {
            if (error) std::rethrow_exception(error);
        }

        const int network = static_cast<int>(work[0].kind == LeafKind::Network)
            + static_cast<int>(work[1].kind == LeafKind::Network);
        const int collisions =
            static_cast<int>(work[0].kind == LeafKind::Collision)
            + static_cast<int>(work[1].kind == LeafKind::Collision);
        require_test(network == 1, "leaf had multiple owners");
        require_test(collisions == 1, "losing collector was not a collision");
        require_test(root.children.empty(), "collision published children");
        require_test(root.state.load() == NodeState::Pending,
                     "network leaf did not remain Pending");
    }

    require_test(root.n_in_flight.load() == 0,
                 "collision test leaked a path reservation");
    require_test(root.state.load() == NodeState::Unexpanded,
                 "uncommitted network leaf stayed Pending");
}

void test_tt_children_are_published_atomically_and_without_backup() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard board = startup_board();
    const std::vector<int> legal = board.getLegalMoveIndices();
    MCTSTestAccess::store(mcts, board, legal, uniform_policy(), 0.25f);

    MCTSNode root(0.0f);
    WorkerContext owner_context;
    owner_context.board = board;
    WorkerContext observer_context;
    observer_context.board = board;
    std::atomic<bool> cancelled{false};
    PublicationGate gate;
    WaveTestHooks hooks;
    hooks.before_tt_publish = [&](MCTSNode* node) {
        require_test(node == &root, "unexpected TT publication node");
        gate.block_owner();
    };

    LeafWork owner;
    std::exception_ptr owner_error;
    std::thread owner_thread([&]() {
        try {
            owner = MCTSTestAccess::collect_leaf(
                mcts, &root, owner_context, 1.4f, cancelled, &hooks);
        }
        catch (...) {
            owner_error = std::current_exception();
        }
    });

    gate.wait_until_blocked();
    require_test(root.state.load() == NodeState::Pending,
                 "TT node was visible before publication");
    require_test(root.children.empty(),
                 "partial TT children were visible while Pending");

    LeafWork observer = MCTSTestAccess::collect_leaf(
        mcts, &root, observer_context, 1.4f, cancelled);
    require_test(observer.kind == LeafKind::Collision,
                 "observer traversed a Pending TT node");

    gate.release();
    owner_thread.join();
    if (owner_error) std::rethrow_exception(owner_error);

    require_test(root.state.load() == NodeState::Expanded,
                 "TT node was not published Expanded");
    require_test(root.children.size() == legal.size(),
                 "TT publication lost legal children");
    require_test(owner.kind == LeafKind::Network && owner.node != &root,
                 "TT hit did not continue to a descendant");
    Chessboard replay = board;
    for (int move_idx : path_to(&root, owner.node)) {
        require_test(mcts.apply_move_by_index(replay, move_idx),
                     "could not replay path after TT publication");
    }
    std::vector<float> expected_tensor;
    replay.getAlphaZeroTensor(expected_tensor);
    require_same_tensor(owner.tensor, expected_tensor);
    require_test(root.visit_count == 0 && root.total_value == 0.0f,
                 "TT publication performed a backup");
    for (const auto& child : root.children) {
        require_test(child.second->visit_count == 0,
                     "TT publication changed child visits");
    }

    owner = LeafWork{};
    observer = LeafWork{};
    require_test(root.n_in_flight.load() == 0,
                 "TT publication leaked reservations");

    WorkerContext after_context;
    after_context.board = board;
    LeafWork after = MCTSTestAccess::collect_leaf(
        mcts, &root, after_context, 1.4f, cancelled);
    require_test(after.kind == LeafKind::Network && after.node != &root,
                 "published TT node was not traversable");
    require_test(root.visit_count == 0,
                 "post-publication descent performed a backup");
}

void test_rule50_mismatch_misses_and_terminal_skips_cache() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard fresh;
    fresh.loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    MCTSTestAccess::store(
        mcts, fresh, fresh.getLegalMoveIndices(), uniform_policy(), 0.5f);

    Chessboard old;
    old.loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 1");
    MCTSNode old_root(0.0f);
    WorkerContext old_context;
    old_context.board = old;
    std::atomic<bool> cancelled{false};
    mcts.reset_counters();
    {
        LeafWork work = MCTSTestAccess::collect_leaf(
            mcts, &old_root, old_context, 1.4f, cancelled);
        require_test(work.kind == LeafKind::Network,
                     "rule50 mismatch reused a cached evaluation");
    }
    require_test(mcts.get_counters().tt_rule50_rejects == 1,
                 "rule50 mismatch was not classified");

    Chessboard terminal;
    terminal.loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1");
    MCTSTestAccess::store(
        mcts, terminal, terminal.getLegalMoveIndices(),
        uniform_policy(), 0.75f);
    MCTSNode terminal_root(0.0f);
    WorkerContext terminal_context;
    terminal_context.board = terminal;
    mcts.reset_counters();
    {
        LeafWork work = MCTSTestAccess::collect_leaf(
            mcts, &terminal_root, terminal_context, 1.4f, cancelled);
        require_test(work.kind == LeafKind::Terminal,
                     "rule terminal was not returned as Terminal");
        require_test(work.terminal_value == 0.0f,
                     "rule50 terminal value was not a draw");
    }
    const SearchCounters counters = mcts.get_counters();
    require_test(counters.tt_hits == 0 && counters.tt_misses == 0
                     && counters.tt_position_matches == 0,
                 "terminal position probed the evaluation cache");
    require_test(terminal_root.state.load() == NodeState::Terminal,
                 "terminal node state was not published");
}

void test_exception_restores_board_and_reservations() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard board = startup_board();
    auto root = expanded_root(mcts, board, 16);
    WorkerContext context;
    context.board = board;
    const std::string fen_before = context.board.toFEN();
    const std::size_t history_before = context.board.getMoveHistory().size();
    std::atomic<bool> cancelled{false};
    WaveTestHooks hooks;
    hooks.before_claim = [&](MCTSNode* node) {
        require_test(node != root.get(),
                     "expanded root was claimed unexpectedly");
        throw std::runtime_error("forced collection failure");
    };

    bool failed = false;
    try {
        (void)MCTSTestAccess::collect_leaf(
            mcts, root.get(), context, 1.4f, cancelled, &hooks);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }

    require_test(failed, "forced collection exception was not propagated");
    require_test(context.board.toFEN() == fen_before,
                 "exception did not restore worker board");
    require_test(context.board.getMoveHistory().size() == history_before,
                 "exception did not restore worker history");
    require_test(root->n_in_flight.load() == 0,
                 "exception leaked root reservation");
    for (const auto& child : root->children) {
        require_test(child.second->n_in_flight.load() == 0,
                     "exception leaked child reservation");
        require_test(child.second->state.load() == NodeState::Unexpanded,
                     "exception changed an unclaimed child state");
    }
}

std::unique_ptr<MCTSNode> expanded_root(
        MCTS& mcts, Chessboard& board, int visits) {
    auto root = std::make_unique<MCTSNode>(0.0f);
    mcts.expand_node_single(root.get(), board);
    root->visit_count = visits;
    return root;
}

void test_wave_fills_beyond_worker_count_and_respects_budget() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    auto root = expanded_root(mcts, board, 64);
    SearchExecutor executor;
    std::vector<WorkerContext> contexts(2);
    for (auto& context : contexts) context.board = board;
    std::atomic<bool> cancelled{false};

    {
        std::vector<LeafWork> work = MCTSTestAccess::collect_wave(
            mcts, root.get(), 1.4f, 8, 2, executor, contexts, cancelled);
        require_test(work.size() > 2,
                     "workers produced at most one leaf each");
        require_test(work.size() <= 8, "wave exceeded its slot budget");
    }
    require_test(root->n_in_flight.load() == 0,
                 "filled wave leaked root reservations");

    for (auto& context : contexts) context.board = board;
    {
        std::vector<LeafWork> work = MCTSTestAccess::collect_wave(
            mcts, root.get(), 1.4f, 3, 2, executor, contexts, cancelled);
        require_test(work.size() == 3,
                     "wave did not honor an exact final budget of three");
    }
}

void test_single_path_returns_a_partial_wave() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 64, 0);
    Chessboard board = startup_board();
    const int only_move = board.getLegalMoveIndices().front();
    MCTSNode root(0.0f);
    root.visit_count = 8;
    root.children.emplace_back(
        only_move, std::make_unique<MCTSNode>(1.0f, only_move, &root));
    root.state.store(NodeState::Expanded);

    SearchExecutor executor;
    std::vector<WorkerContext> contexts(2);
    for (auto& context : contexts) context.board = board;
    std::atomic<bool> cancelled{false};
    {
        std::vector<LeafWork> work = MCTSTestAccess::collect_wave(
            mcts, &root, 1.4f, 8, 2, executor, contexts, cancelled);
        require_test(work.size() == 1,
                     "single path did not produce a partial wave");
        require_test(work.front().kind == LeafKind::Network,
                     "single path did not reach its network leaf");
    }
    require_test(root.n_in_flight.load() == 0,
                 "partial wave leaked root reservation");
    require_test(root.children.front().second->state.load()
                     == NodeState::Unexpanded,
                 "partial wave left its leaf Pending");
}

void test_captured_tensor_matches_replayed_real_history() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    require_test(board.movePieceUCI("e2e4"), "e2e4 failed");
    require_test(board.movePieceUCI("e7e5"), "e7e5 failed");
    require_test(board.movePieceUCI("g1f3"), "g1f3 failed");
    const std::string fen_before = board.toFEN();
    const std::size_t history_before = board.getMoveHistory().size();
    auto root = expanded_root(mcts, board, 64);

    SearchExecutor executor;
    std::vector<WorkerContext> contexts(2);
    for (auto& context : contexts) context.board = board;
    std::atomic<bool> cancelled{false};
    std::vector<LeafWork> work = MCTSTestAccess::collect_wave(
        mcts, root.get(), 1.4f, 4, 2, executor, contexts, cancelled);
    require_test(!work.empty(), "history fixture produced no network leaf");

    for (const LeafWork& leaf : work) {
        require_test(leaf.kind == LeafKind::Network,
                     "history fixture returned a non-network result");
        Chessboard replay = board;
        for (int move_idx : path_to(root.get(), leaf.node)) {
            require_test(mcts.apply_move_by_index(replay, move_idx),
                         "could not replay captured path");
        }
        std::vector<float> expected;
        replay.getAlphaZeroTensor(expected);
        require_same_tensor(leaf.tensor, expected);
    }

    for (const auto& context : contexts) {
        require_test(context.board.toFEN() == fen_before,
                     "worker board was not restored");
        require_test(context.board.getMoveHistory().size() == history_before,
                     "worker history was not restored");
    }
}

void require_no_pending_result(const std::vector<WorkerContext>& contexts) {
    for (const WorkerContext& context : contexts) {
        require_test(context.results.empty(),
                     "a worker context kept a collected result");
    }
}

void require_no_residue(const MCTSNode* node) {
    require_test(node->n_in_flight.load() == 0,
                 "a node kept in-flight units");
    require_test(node->state.load() != NodeState::Pending,
                 "a node stayed Pending");
    for (const auto& child : node->children) {
        require_no_residue(child.second.get());
    }
}

void test_merge_failure_releases_every_collected_reservation() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_tuning(SearchTuning{3, 0.30f, 4});
    Chessboard board = startup_board();
    auto root = expanded_root(mcts, board, 64);
    SearchExecutor executor;
    std::vector<WorkerContext> contexts(2);
    for (auto& context : contexts) context.board = board;
    std::atomic<bool> cancelled{false};
    WaveTestHooks hooks;
    hooks.before_merge = [&]() {
        const std::size_t collected =
            contexts[0].results.size() + contexts[1].results.size();
        require_test(collected >= 2,
                     "merge fixture collected fewer than two leaves");
        throw std::bad_alloc();
    };

    bool failed = false;
    try {
        (void)MCTSTestAccess::collect_wave(
            mcts, root.get(), 1.4f, 4, 2, executor, contexts, cancelled,
            &hooks);
    }
    catch (const std::bad_alloc&) {
        failed = true;
    }

    require_test(failed, "merge failure was swallowed");
    require_no_pending_result(contexts);
    require_no_residue(root.get());
    for (const WorkerContext& context : contexts) {
        require_test(context.board.toFEN() == startup_board().toFEN(),
                     "merge failure did not restore a worker board");
    }
}

void test_partial_transfer_failure_releases_both_owners() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_tuning(SearchTuning{3, 0.30f, 4});
    Chessboard board = startup_board();
    auto root = expanded_root(mcts, board, 64);
    SearchExecutor executor;
    std::vector<WorkerContext> contexts(2);
    for (auto& context : contexts) context.board = board;
    std::atomic<bool> cancelled{false};
    std::size_t transferts = 0;
    WaveTestHooks hooks;
    hooks.after_transfer = [&](std::size_t index) {
        ++transferts;
        if (index == 0) throw std::bad_alloc();
    };

    bool failed = false;
    try {
        (void)MCTSTestAccess::collect_wave(
            mcts, root.get(), 1.4f, 4, 2, executor, contexts, cancelled,
            &hooks);
    }
    catch (const std::bad_alloc&) {
        failed = true;
    }

    require_test(failed, "partial transfer failure was swallowed");
    require_test(transferts == 1,
                 "partial transfer did not stop at the first result");
    require_no_pending_result(contexts);
    require_no_residue(root.get());
}

void test_merge_failure_through_step_analysis_recovers() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    WaveTestHooks hooks;
    hooks.before_merge = []() { throw std::bad_alloc(); };
    MCTSTestAccess::set_wave_hooks(mcts, &hooks);

    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 4);
    }
    catch (const std::bad_alloc&) {
        failed = true;
    }

    require_test(failed, "merge failure through step_analysis was swallowed");
    const TreeReport rest = mcts.inspect_tree();
    require_test(rest.en_vol == 0 && rest.pending == 0
                     && rest.violations == 0,
                 "step_analysis left residue after merge failure");
    require_test(board.toFEN() == startup_board().toFEN(),
                 "step_analysis changed the input board");

    MCTSTestAccess::set_wave_hooks(mcts, nullptr);
    mcts.reset_analysis();
    mcts.step_analysis(board, 3, 1.4f, 3, 2);
    require_test(MCTSTestAccess::pending_results(mcts) == 0,
                 "contexts kept results across recovery");
    const TreeReport recovered = mcts.inspect_tree();
    require_test(recovered.en_vol == 0 && recovered.pending == 0
                     && recovered.violations == 0,
                 "recovery search left residue");
    require_test(recovered.root_visits == 3,
                 "recovery search did not complete its budget");
    const std::vector<int> legal = board.getLegalMoveIndices();
    mcts.update_root(legal.front());
}

void test_merge_failure_in_mcts_search_keeps_instance_usable() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    WaveTestHooks hooks;
    hooks.before_merge = []() { throw std::bad_alloc(); };
    MCTSTestAccess::set_wave_hooks(mcts, &hooks);

    bool failed = false;
    try {
        (void)mcts.mcts_search(board, 8, 1.4f, false, 8, 4);
    }
    catch (const std::bad_alloc&) {
        failed = true;
    }

    require_test(failed, "merge failure in mcts_search was swallowed");
    // La racine locale de mcts_search est detruite pendant la propagation :
    // les contextes ne doivent plus posseder un seul resultat, meme sans
    // arbre a inspecter.
    require_test(MCTSTestAccess::pending_results(mcts) == 0,
                 "mcts_search merge failure left owned results in contexts");
    require_test(board.toFEN() == startup_board().toFEN(),
                 "mcts_search merge failure changed the input board");

    MCTSTestAccess::set_wave_hooks(mcts, nullptr);
    const std::vector<float> policy =
        mcts.mcts_search(board, 3, 1.4f, false, 3, 2);
    require_test(!policy.empty(), "mcts_search did not recover");
    require_test(MCTSTestAccess::pending_results(mcts) == 0,
                 "recovery left results in contexts");
}

}  // namespace

int main() {
    try {
        test_only_one_collector_claims_a_leaf();
        test_tt_children_are_published_atomically_and_without_backup();
        test_rule50_mismatch_misses_and_terminal_skips_cache();
        test_exception_restores_board_and_reservations();
        test_wave_fills_beyond_worker_count_and_respects_budget();
        test_single_path_returns_a_partial_wave();
        test_captured_tensor_matches_replayed_real_history();
        test_merge_failure_releases_every_collected_reservation();
        test_partial_transfer_failure_releases_both_owners();
        test_merge_failure_through_step_analysis_recovers();
        test_merge_failure_in_mcts_search_keeps_instance_usable();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
