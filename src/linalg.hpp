#pragma once

#include <concepts>

namespace linalg {
    using size_type = std::size_t;

    template <typename T>
    concept NumberLike = std::integral<T> || std::floating_point<T>;

    template<NumberLike Number>
    using DynamicMatrix = Matrix<linalg::dynamic, linalg::dynamic, Number>;
} // namespace linalg
