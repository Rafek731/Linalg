#pragma once

#include <concepts>

namespace linalg {
    template <typename T>
    concept Number = std::integral<T> || std::floating_point<T>;
} // namespace linalg
