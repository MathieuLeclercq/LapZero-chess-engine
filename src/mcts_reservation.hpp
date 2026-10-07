#pragma once

#include <cstdint>
#include <vector>

struct MCTSNode;

enum class NodeState : std::uint8_t {
    Unexpanded,
    Pending,
    Expanded,
    Terminal,
};

class PathReservation {
public:
    PathReservation() = default;
    PathReservation(PathReservation&& other) noexcept;
    PathReservation& operator=(PathReservation&& other) noexcept;
    PathReservation(const PathReservation&) = delete;
    PathReservation& operator=(const PathReservation&) = delete;
    ~PathReservation();

    void reserve(MCTSNode* node, std::uint32_t units = 1);
    bool try_claim(MCTSNode* node);
    void publish(NodeState state) noexcept;
    void release() noexcept;

private:
    // Une entree par noeud reserve, avec le nombre exact d'unites posees a cet
    // appel : les liberer par une constante globale desynchroniserait les
    // compteurs des que l'amplitude du virtual loss depasse 1.
    struct Reservation {
        MCTSNode* node;
        std::uint32_t units;
    };

    std::vector<Reservation> m_path;
    MCTSNode* m_owned_pending = nullptr;
};
