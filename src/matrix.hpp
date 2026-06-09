#pragma once

#include <cstddef>
#include <array>
#include <algorithm> // std::transform
#include <execution> // std::excecution::unseq
#include <functional> // std::plus, std::minus

#include "number.hpp"

namespace linalg {
using size_type = std::size_t;

static constexpr size_type dynamic = static_cast<size_type>(-1); 

template <size_type _rows, size_type _cols, Number NumberLike = double>
class Matrix {
    static_assert(_rows > 0 && _cols > 0, "Matrix dimensions (_rows and _cols) must be greater than 0.");
public:

    Matrix();
    explicit Matrix(NumberLike init_value);

    [[nodiscard]] static constexpr size_type rows() noexcept { return _rows; }
    [[nodiscard]] static constexpr size_type cols() noexcept { return _cols; }
    [[nodiscard]] [[nodiscard]] static constexpr size_type size() noexcept { return size_; }

    /// arithmetic operators
    [[nodiscard]] Matrix  add (const NumberLike scalar)       const noexcept;
    [[nodiscard]] Matrix  add (const Matrix& other)           const noexcept;
    [[nodiscard]] Matrix  operator+ (const NumberLike scalar) const noexcept;
    [[nodiscard]] Matrix  operator+ (const Matrix& other)     const noexcept;
    
    Matrix& add_inplace(const NumberLike scalar) noexcept;
    Matrix& add_inplace(const Matrix& other)     noexcept;
    Matrix& operator+= (const NumberLike scalar) noexcept;
    Matrix& operator+= (const Matrix& other)     noexcept;

    [[nodiscard]] Matrix  subtract (const NumberLike scalar) const noexcept;
    [[nodiscard]] Matrix  subtract (const Matrix& other)     const noexcept;
    [[nodiscard]] Matrix  operator-(const NumberLike scalar) const noexcept;
    [[nodiscard]] Matrix  operator-(const Matrix& other)     const noexcept;
    
    Matrix& subtract_inplace(const NumberLike scalar) noexcept;
    Matrix& subtract_inplace(const Matrix& other)     noexcept;
    Matrix& operator-=(const NumberLike scalar)       noexcept;
    Matrix& operator-=(const Matrix& other)           noexcept;

    [[nodiscard]] Matrix  mul (const NumberLike scalar)  const noexcept;
    Matrix& mul_inplace(const NumberLike scalar) noexcept;
    
    template<size_type _other_cols>
    [[nodiscard]] Matrix<_rows, _other_cols, NumberLike> mul(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept;
    template<size_type _other_cols>
    [[nodiscard]] Matrix<_rows, _other_cols, NumberLike> operator*(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept;
    
    [[nodiscard]] Matrix  operator* (const NumberLike scalar) const noexcept;
    Matrix& operator*=(const NumberLike scalar) noexcept;

    [[nodiscard]] Matrix  divide (const NumberLike scalar)        const noexcept;
    Matrix& divide_inplace(const NumberLike scalar) noexcept;
    [[nodiscard]] Matrix  operator/ (const NumberLike scalar)     const noexcept;
    Matrix& operator/=(const NumberLike scalar)     noexcept;

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

    NumberLike& operator[](size_type row, size_type col) noexcept;
    NumberLike operator[](size_type row, size_type col) const noexcept;
private:
    static constexpr size_type size_ = _rows * _cols;
    std::array<NumberLike, size_> data_;

    // Allows operators to access data_ across different sizes/types
    template <size_type _r, size_type _c, Number _N>
    friend class Matrix;
};

// --- Constructors ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>::Matrix() : data_{}
{}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>::Matrix(NumberLike init_value) : data_{} {
    data_.fill(init_value);
}

// --- Addition ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::add(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, NumberLike> tmp;
    
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   tmp.data_.begin(),
                   [scalar](const NumberLike val){ return val + scalar; });
    return tmp;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::add(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, NumberLike> tmp;

    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(), 
                   tmp.data_.begin(), 
                   std::plus<>());

    return tmp;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator+(const NumberLike scalar) const noexcept {
    return add(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator+(const Matrix& other) const noexcept {
    return add(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::add_inplace(const NumberLike scalar) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const NumberLike val){ return val + scalar; });
    
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::add_inplace(const Matrix& other) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   data_.begin(),
                   std::plus<>());
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator+=(const NumberLike scalar) noexcept {
    return add_inplace(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator+=(const Matrix& other) noexcept {
    return add_inplace(other);
}

// --- Subtraction ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::subtract(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, NumberLike> tmp;
    
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   tmp.data_.begin(),
                   [scalar](const NumberLike val){ return val - scalar; });
    return tmp;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::subtract(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, NumberLike> tmp;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(), 
                   tmp.data_.begin(), 
                   std::minus<>());
    return tmp;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator-(const NumberLike scalar) const noexcept {
    return subtract(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator-(const Matrix& other) const noexcept {
    return subtract(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::subtract_inplace(const NumberLike scalar) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const NumberLike val){ return val - scalar; });
    
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::subtract_inplace(const Matrix& other) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   data_.begin(),
                   std::minus<>());
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator-=(const NumberLike scalar) noexcept {
    return subtract_inplace(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator-=(const Matrix& other) noexcept {
    return subtract_inplace(other);
}

// --- Multiplication ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, NumberLike> tmp;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   tmp.data_.begin(), 
                   [scalar](const NumberLike val){ return val * scalar; });
    return tmp;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::mul_inplace(const NumberLike scalar) noexcept {
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const NumberLike val){ return val * scalar; });
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
template<size_type _other_cols>
Matrix<_rows, _other_cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
template<size_type _other_cols>
Matrix<_rows, _other_cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator*(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept {
    return mul(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator*(const NumberLike scalar) const noexcept {
    return mul(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator*=(const NumberLike scalar) noexcept {
    return mul_inplace(scalar);
}

// --- Division ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::divide(const NumberLike scalar) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::divide_inplace(const NumberLike scalar) noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator/(const NumberLike scalar) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator/=(const NumberLike scalar) noexcept {
}

// --- Logical Operators ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::negate() const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!() const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::eq(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::neq(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::lt(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::le(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::gt(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::ge(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator==(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!=(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<=(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>(const Matrix& other) const noexcept {
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>=(const Matrix& other) const noexcept {
}

} // namespace linalg