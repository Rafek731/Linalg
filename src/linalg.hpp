#pragma once

#include <concepts>

namespace linalg {
    template <typename T>
    concept NumberLike = std::integral<T> || std::floating_point<T>;
} // namespace linalg
