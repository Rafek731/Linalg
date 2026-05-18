#pragma once

#include <concepts>

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;