#pragma once

#include <cstddef>
#include <array>
#include "number.hpp"

namespace linalg {

static constexpr std::size_t dynamic = static_cast<std::size_t>(-1); 

template <std::size_t _rows, std::size_t _cols, Number NumberLike = double>
class Matrix {
public:

    Matrix();
    explicit Matrix(NumberLike init_value);

    static constexpr std::size_t rows() noexcept { return _rows; }
    static constexpr std::size_t cols() noexcept { return _cols; }
    static constexpr std::size_t size() noexcept { return size_; }

    /// arithmetic operators
    Matrix  add (const NumberLike scalar)   const noexcept;
    Matrix  add (const Matrix& other)       const noexcept;
    Matrix  operator+ (const NumberLike scalar) const noexcept;
    Matrix  operator+ (const Matrix& other)     const noexcept;
    
    Matrix& add_inplace(const NumberLike scalar)    noexcept;
    Matrix& add_inplace(const Matrix& other)        noexcept;
    Matrix& operator+= (const NumberLike scalar)     noexcept;
    Matrix& operator+= (const Matrix& other)         noexcept;

    Matrix  subtract (const NumberLike scalar)  const noexcept;
    Matrix  subtract (const Matrix& other)      const noexcept;
    Matrix  operator-(const NumberLike scalar) const noexcept;
    Matrix  operator-(const Matrix& other)     const noexcept;
    
    Matrix& subtract_inplace(const NumberLike scalar) noexcept;
    Matrix& subtract_inplace(const Matrix& other)     noexcept;
    Matrix& operator-=(const NumberLike scalar) noexcept;
    Matrix& operator-=(const Matrix& other)     noexcept;

    Matrix  mul (const NumberLike scalar) const noexcept;
    Matrix& mul_inplace(const NumberLike scalar) noexcept;
    
    template<std::size_t _other_cols>
    Matrix<_rows, _other_cols, NumberLike> mul(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept;
    template<std::size_t _other_cols>
    Matrix<_rows, _other_cols, NumberLike> operator*(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept;
    
    Matrix  operator* (const NumberLike scalar) const noexcept;
    Matrix& operator*=(const NumberLike scalar) noexcept;

    Matrix  divide (const NumberLike scalar) const noexcept;
    Matrix& divide_inplace(const NumberLike scalar) noexcept;
    Matrix  operator/ (const NumberLike scalar) const noexcept;
    Matrix& operator/=(const NumberLike scalar) noexcept;

    Matrix<_rows, _cols, bool> negate() const noexcept;
    Matrix<_rows, _cols, bool> operator!() const noexcept;

    Matrix<_rows, _cols, bool> eq (const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> neq(const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> lt (const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> le (const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> gt (const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> ge (const Matrix& other) const noexcept;

    Matrix<_rows, _cols, bool> operator==(const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> operator!=(const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> operator< (const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> operator<=(const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> operator> (const Matrix& other) const noexcept;
    Matrix<_rows, _cols, bool> operator>=(const Matrix& other) const noexcept;

private:
    static constexpr std::size_t size_ = _rows * _cols;
    std::array<NumberLike, size_> data_;

    // Allows operators to access data_ across different sizes/types
    template <std::size_t r, std::size_t c, Number N>
    friend class Matrix;
};

// --- Constructors ---

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>::Matrix() {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>::Matrix(NumberLike init_value) {
}

// --- Addition ---

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::add(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::add(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator+(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator+(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::add_inplace(const NumberLike scalar) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::add_inplace(const Matrix& other) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator+=(const NumberLike scalar) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator+=(const Matrix& other) noexcept {
}

// --- Subtraction ---

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::subtract(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::subtract(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator-(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator-(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::subtract_inplace(const NumberLike scalar) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::subtract_inplace(const Matrix& other) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator-=(const NumberLike scalar) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator-=(const Matrix& other) noexcept {
}

// --- Multiplication ---

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::mul_inplace(const NumberLike scalar) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
template<std::size_t _other_cols>
Matrix<_rows, _other_cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
template<std::size_t _other_cols>
Matrix<_rows, _other_cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator*(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator*(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator*=(const NumberLike scalar) noexcept {
}

// --- Division ---

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::divide(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::divide_inplace(const NumberLike scalar) noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator/(const NumberLike scalar) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator/=(const NumberLike scalar) noexcept {
}

// --- Logical Operators ---

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::negate() const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!() const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::eq(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::neq(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::lt(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::le(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::gt(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::ge(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator==(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!=(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<=(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>(const Matrix& other) const noexcept {
}

template<std::size_t _rows, std::size_t _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>=(const Matrix& other) const noexcept {
}

} // namespace linalg