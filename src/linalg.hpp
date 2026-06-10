#pragma once

#include <concepts>

namespace linalg {
    using size_type = std::size_t;
    
    template <typename T>
    concept NumberLike = std::integral<T> || std::floating_point<T>;
} // namespace linalg
