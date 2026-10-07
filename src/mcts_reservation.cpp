#include "mcts_reservation.hpp"

#include "mcts.hpp"

#include <atomic>
#include <utility>

PathReservation::PathReservation(PathReservation&& other) noexcept
    : m_path(std::move(other.m_path)),
      m_owned_pending(std::exchange(other.m_owned_pending, nullptr)) {
    other.m_path.clear();
}

PathReservation& PathReservation::operator=(PathReservation&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    release();
    m_path = std::move(other.m_path);
    m_owned_pending = std::exchange(other.m_owned_pending, nullptr);
    other.m_path.clear();
    return *this;
}

PathReservation::~PathReservation() {
    release();
}

void PathReservation::reserve(MCTSNode* node, std::uint32_t units) {
    if (node == nullptr || units == 0) {
        return;
    }
    // Enregistrer d'abord l'entree : si le vecteur ne peut pas grandir,
    // aucun compteur n'a encore ete modifie.
    m_path.push_back(Reservation{node, units});
    node->n_in_flight.fetch_add(units, std::memory_order_relaxed);
}

bool PathReservation::try_claim(MCTSNode* node) {
    if (node == nullptr || m_owned_pending != nullptr) {
        return false;
    }
    NodeState expected = NodeState::Unexpanded;
    if (!node->state.compare_exchange_strong(
            expected, NodeState::Pending,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    m_owned_pending = node;
    return true;
}

void PathReservation::publish(NodeState state) noexcept {
    if (m_owned_pending == nullptr) {
        return;
    }
    m_owned_pending->state.store(state, std::memory_order_release);
    m_owned_pending = nullptr;
}

void PathReservation::release() noexcept {
    if (m_owned_pending != nullptr) {
        NodeState expected = NodeState::Pending;
        m_owned_pending->state.compare_exchange_strong(
            expected, NodeState::Unexpanded,
            std::memory_order_release, std::memory_order_relaxed);
        m_owned_pending = nullptr;
    }

    for (const Reservation& reservation : m_path) {
        reservation.node->n_in_flight.fetch_sub(
            reservation.units, std::memory_order_relaxed);
    }
    m_path.clear();
}
