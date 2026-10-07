#include "evaluation_cache.hpp"
#include "test_support.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

class StartGate {
public:
    explicit StartGate(int participants) : m_remaining(participants) {}

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

EvaluationCacheKey key_for_hash(std::uint64_t hash) {
    EvaluationCacheKey key;
    key.position_hash = hash;
    key.current_context_hash = hash + 100;
    key.history_hash = hash + 200;
    key.combined_hash = hash + 300;
    key.half_move_clock = 4;
    return key;
}

std::vector<float> policy_with(const std::vector<int>& legal,
                               float base) {
    std::vector<float> policy(4672, 0.0f);
    for (std::size_t i = 0; i < legal.size(); ++i) {
        policy[legal[i]] = base + static_cast<float>(i) * 0.01f;
    }
    return policy;
}

bool probe_matches(const TTProbe& probe, const std::vector<int>& legal,
                   const std::vector<float>& policy, float value) {
    if (probe.status != TTProbeStatus::HIT || probe.value != value
        || probe.policy_size != static_cast<int>(legal.size())) {
        return false;
    }
    for (int i = 0; i < probe.policy_size; ++i) {
        if (probe.legal_policy[i].first != legal[i]
            || probe.legal_policy[i].second != policy[legal[i]]) {
            return false;
        }
    }
    return true;
}

void test_size_and_stripe_count_are_bounded() {
    bool zero_rejected = false;
    try {
        EvaluationCache invalid(0, 0);
    }
    catch (const std::invalid_argument&) {
        zero_rejected = true;
    }
    require_test(zero_rejected, "zero-sized cache was accepted");

    EvaluationCache small(3, 0);
    EvaluationCache large(10000, 0);
    require_test(small.size() == 3, "cache size changed");
    require_test(small.stripe_count() == 3, "small stripe count mismatch");
    require_test(large.stripe_count() == 4096,
                 "stripe count was not capped");
}

void test_probe_is_a_stable_snapshot() {
    EvaluationCache cache(3, 0);
    const std::vector<int> legal = {5, 17, 42};
    const auto policy_a = policy_with(legal, 0.10f);
    const auto policy_b = policy_with(legal, 0.70f);
    const EvaluationCacheKey key_a = key_for_hash(1);
    const EvaluationCacheKey key_b = key_for_hash(4);

    cache.store(key_a, legal, policy_a.data(), 0.25f);
    const TTProbe saved = cache.probe(key_a);
    cache.store(key_b, legal, policy_b.data(), -0.75f);

    require_test(probe_matches(saved, legal, policy_a, 0.25f),
                 "saved snapshot changed after overwrite");
    require_test(cache.probe(key_a).status == TTProbeStatus::MISS,
                 "colliding hash was accepted");
    require_test(probe_matches(cache.probe(key_b), legal, policy_b, -0.75f),
                 "replacement entry is incomplete");
}

void test_clear_forgets_stored_entries_but_not_snapshots() {
    EvaluationCache cache(5, 0);
    const std::vector<int> legal = {5, 17, 42};
    const auto policy = policy_with(legal, 0.30f);
    const EvaluationCacheKey key = key_for_hash(9);
    cache.store(key, legal, policy.data(), 0.75f);
    const TTProbe saved = cache.probe(key);
    require_test(saved.status == TTProbeStatus::HIT,
                 "store then probe did not hit before clear");

    cache.clear();

    require_test(cache.probe(key).status == TTProbeStatus::MISS,
                 "clear left an old hit");
    require_test(probe_matches(saved, legal, policy, 0.75f),
                 "snapshot taken before clear was invalidated");
}

void test_concurrent_collisions_never_mix_entries() {
    EvaluationCache cache(3, 0);
    const std::vector<int> legal = {5, 17, 42, 99};
    const auto policy_a = policy_with(legal, 0.10f);
    const auto policy_b = policy_with(legal, 0.70f);
    const EvaluationCacheKey key_a = key_for_hash(1);
    const EvaluationCacheKey key_b = key_for_hash(4);
    StartGate gate(4);
    std::atomic<int> failures{0};
    std::atomic<int> hits{0};

    auto writer = [&](const EvaluationCacheKey& key,
                      const std::vector<float>& policy, float value) {
        gate.wait();
        for (int i = 0; i < 20000; ++i) {
            cache.store(key, legal, policy.data(), value);
        }
    };
    auto reader = [&](int offset) {
        gate.wait();
        for (int i = 0; i < 40000; ++i) {
            const bool use_a = ((i + offset) % 2) == 0;
            const TTProbe probe = cache.probe(use_a ? key_a : key_b);
            if (probe.status == TTProbeStatus::MISS) {
                continue;
            }
            hits.fetch_add(1, std::memory_order_relaxed);
            const bool valid = use_a
                ? probe_matches(probe, legal, policy_a, 0.25f)
                : probe_matches(probe, legal, policy_b, -0.75f);
            if (!valid) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::thread writer_a(writer, std::cref(key_a), std::cref(policy_a), 0.25f);
    std::thread writer_b(writer, std::cref(key_b), std::cref(policy_b), -0.75f);
    std::thread reader_a(reader, 0);
    std::thread reader_b(reader, 1);
    writer_a.join();
    writer_b.join();
    reader_a.join();
    reader_b.join();

    require_test(hits.load() > 0, "concurrent test observed no cache hit");
    require_test(failures.load() == 0, "concurrent snapshot was torn");
}

void test_rejection_order_and_contexts() {
    Chessboard at_zero;
    Chessboard at_ninety_nine;
    at_zero.loadFEN("8/8/8/8/8/2k5/8/R3K3 w - - 0 1");
    at_ninety_nine.loadFEN("8/8/8/8/8/2k5/8/R3K3 w - - 99 1");
    const EvaluationCacheKey zero_key = at_zero.getEvaluationCacheKey(0);
    const EvaluationCacheKey rule50_key =
        at_ninety_nine.getEvaluationCacheKey(0);
    const std::vector<int> legal = at_zero.getLegalMoveIndices();
    const auto policy = policy_with(legal, 0.20f);
    EvaluationCache cache(17, 0);
    cache.store(zero_key, legal, policy.data(), 0.5f);

    require_test(cache.probe(rule50_key).status
                     == TTProbeStatus::RULE50_REJECT,
                 "rule-50 mismatch was not rejected first");

    EvaluationCacheKey context_key = zero_key;
    context_key.current_context_hash ^= 1;
    context_key.combined_hash ^= 1;
    require_test(cache.probe(context_key).status
                     == TTProbeStatus::CONTEXT_REJECT,
                 "current context mismatch was not rejected");

    EvaluationCacheKey history_key = zero_key;
    history_key.history_hash ^= 1;
    history_key.combined_hash ^= 1;
    require_test(cache.probe(history_key).status
                     == TTProbeStatus::HISTORY_REJECT,
                 "history mismatch was not rejected");

    EvaluationCacheKey missing_key = zero_key;
    missing_key.position_hash += 17;
    require_test(cache.probe(missing_key).status == TTProbeStatus::MISS,
                 "different position hash was not a miss");
}

void test_oversized_policy_is_not_cached() {
    EvaluationCache cache(5, LEGACY_CACHE_HISTORY_DEPTH);
    EvaluationCacheKey stored = key_for_hash(2);
    EvaluationCacheKey changed = stored;
    changed.half_move_clock = 99;
    changed.current_context_hash ^= 1;
    changed.history_hash ^= 1;
    changed.combined_hash ^= 1;

    std::vector<int> trop_longue;
    for (int i = 0; i < TT_MAX_MOVES + 1; ++i) {
        trop_longue.push_back(i);
    }
    const auto politique_longue = policy_with(trop_longue, 0.01f);
    cache.store(stored, trop_longue, politique_longue.data(), 0.125f);
    require_test(cache.probe(stored).status == TTProbeStatus::MISS,
                 "an oversized policy was cached");

    // La frontiere exacte reste stockee et relue integralement.
    std::vector<int> exacte;
    for (int i = 0; i < TT_MAX_MOVES; ++i) {
        exacte.push_back(i);
    }
    const auto politique_exacte = policy_with(exacte, 0.01f);
    cache.store(stored, exacte, politique_exacte.data(), 0.125f);
    const TTProbe probe = cache.probe(changed);
    require_test(probe.status == TTProbeStatus::HIT,
                 "the exact capacity was not cached");
    require_test(probe.policy_size == TT_MAX_MOVES,
                 "the exact capacity was truncated");
}

}  // namespace

int main() {
    try {
        test_size_and_stripe_count_are_bounded();
        test_probe_is_a_stable_snapshot();
        test_clear_forgets_stored_entries_but_not_snapshots();
        test_concurrent_collisions_never_mix_entries();
        test_rejection_order_and_contexts();
        test_oversized_policy_is_not_cached();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
