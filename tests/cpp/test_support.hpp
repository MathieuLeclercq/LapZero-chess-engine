#pragma once

#include <stdexcept>

inline void require_test(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
