#pragma once

#include <cstddef>
#include <array>
#include <algorithm> // std::transform
#include <execution> // std::excecution::unseq
#include <functional> // std::plus, std::minus, std::less, std::less_equal, std::greater, std::greater_equal, std::equal_to, std::not_equal_to
#include <iterator>
#include <limits>  // std::numeric_limits
#include <cassert> // assert()

#include "linalg.hpp"

namespace linalg {

using size_type = std::size_t;

static constexpr size_type dynamic = static_cast<size_type>(-1); 

template <size_type _rows, size_type _cols, NumberLike Number = double>
class Matrix {
    static_assert(_rows > 0 && _cols > 0, "Matrix dimensions (_rows and _cols) must be greater than 0.");
public:

    Matrix() = default;
    explicit Matrix(Number init_value) : Matrix() { data_.fill(init_value); };

    [[nodiscard]] static constexpr size_type rows() noexcept { return _rows; }
    [[nodiscard]] static constexpr size_type cols() noexcept { return _cols; }
    [[nodiscard]] static constexpr size_type size() noexcept { return size_; }

    /// @section arithmetic_operators

    [[nodiscard]] Matrix add (const Number scalar)      const noexcept;
    [[nodiscard]] Matrix add(const Matrix& other)       const noexcept;
    [[nodiscard]] Matrix operator+(const Number scalar) const noexcept;
    [[nodiscard]] Matrix operator+(const Matrix& other) const noexcept;
    
    Matrix& add_inplace(const Number scalar) noexcept;
    Matrix& add_inplace(const Matrix& other) noexcept;
    Matrix& operator+= (const Number scalar) noexcept;
    Matrix& operator+= (const Matrix& other) noexcept;

    [[nodiscard]] Matrix subtract (const Number scalar) const noexcept;
    [[nodiscard]] Matrix subtract (const Matrix& other) const noexcept;
    [[nodiscard]] Matrix operator-(const Number scalar) const noexcept;
    [[nodiscard]] Matrix operator-(const Matrix& other) const noexcept;
    
    Matrix& subtract_inplace(const Number scalar) noexcept;
    Matrix& subtract_inplace(const Matrix& other) noexcept;
    Matrix& operator-=(const Number scalar)       noexcept;
    Matrix& operator-=(const Matrix& other)       noexcept;

    [[nodiscard]] Matrix mul_elements(const Number scalar) const noexcept;
    Matrix& mul_elements_inplace(const Number scalar) noexcept;
    [[nodiscard]] Matrix mul_elements(const Matrix& other) const noexcept;
    Matrix& mul_elements_inplace(const Matrix& other) noexcept;

    template<size_type _other_cols>
    [[nodiscard]] Matrix<_rows, _other_cols, Number> mul_matrix(const Matrix<_cols, _other_cols, Number>& other) const noexcept;
    template<size_type _other_cols>
    [[nodiscard]] Matrix<_rows, _other_cols, Number> operator%(const Matrix<_cols, _other_cols, Number>& other) const noexcept;
    
    [[nodiscard]] Matrix operator*(const Number scalar) const noexcept;
    Matrix& operator*=(const Number scalar) noexcept;
    
    [[nodiscard]] Matrix operator*(const Matrix& other) const noexcept;
    Matrix& operator*=(const Matrix& other) noexcept;

    [[nodiscard]] Matrix divide (const Number scalar)    const noexcept;
    [[nodiscard]] Matrix operator/ (const Number scalar) const noexcept;
    Matrix& divide_inplace(const Number scalar) noexcept;
    Matrix& operator/=(const Number scalar)     noexcept;

    /// @section logical_operators

    [[nodiscard]] Matrix<_rows, _cols, bool> negate() const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!() const noexcept;

    /// @section comparison_operators

    [[nodiscard]] Matrix<_rows, _cols, bool> equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> equal(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> not_equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> not_equal(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> smaller(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> smaller_equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller_equal(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> greater(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> greater(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> greater_equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> greater_equal(const Number scalar) const noexcept;
    

    [[nodiscard]] Matrix<_rows, _cols, bool> operator==(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator==(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator!=(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!=(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator< (const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator< (const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator<=(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator<=(const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator> (const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator> (const Number scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator>=(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator>=(const Number scalar) const noexcept;

    /// @section indexing_operators

    [[nodiscard]] Number& operator[](size_type row, size_type col) noexcept;
    [[nodiscard]] Number  operator[](size_type row, size_type col) const noexcept;

    /// @section iterators
    /// @subsection row_iterator

    // ============================================================================
    // Iterators
    // ============================================================================

    using iterator =        std::array<Number, size_>::iterator;
    using const_iterator =  std::array<Number, size_>::const_iterator;

    /**
     * @struct row_iterator
     * @brief Contiguous iterator for iterating over matrix elements row-by-row.
     * 
     * This iterator provides random-access iteration over matrix elements in row-major order.
     * It maintains a pointer to a matrix element and allows all standard random-access operations.
     */

private:
    static constexpr size_type size_ = _rows * _cols;
    std::array<Number, size_> data_;

    // Allows operators to access data_ across different sizes/types
    template <size_type _r, size_type _c, NumberLike _N>
    friend class Matrix;
};

// --- Addition ---

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::add(const Number scalar) const noexcept {
    Matrix<_rows, _cols, Number> result;
    
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(),
                   [scalar](const Number val){ return val + scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::add(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, Number> result;

    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(), 
                   result.data_.begin(), 
                   std::plus<>());

    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator+(const Number scalar) const noexcept {
    return add(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator+(const Matrix& other) const noexcept {
    return add(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::add_inplace(const Number scalar) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const Number val){ return val + scalar; });
    
    return *this;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::add_inplace(const Matrix& other) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   data_.begin(),
                   std::plus<>());

    return *this;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator+=(const Number scalar) noexcept {
    return add_inplace(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator+=(const Matrix& other) noexcept {
    return add_inplace(other);
}

// --- Subtraction ---

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::subtract(const Number scalar) const noexcept {
    Matrix<_rows, _cols, Number> result;
    
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(),
                   [scalar](const Number val){ return val - scalar; });
    
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::subtract(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, Number> result;

    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(), 
                   result.data_.begin(), 
                   std::minus<>());
    
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator-(const Number scalar) const noexcept {
    return subtract(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator-(const Matrix& other) const noexcept {
    return subtract(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::subtract_inplace(const Number scalar) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const Number val){ return val - scalar; });
    
    return *this;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::subtract_inplace(const Matrix& other) noexcept {
    std::transform(std::execution::unseq,
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   data_.begin(),
                   std::minus<>());
    return *this;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator-=(const Number scalar) noexcept {
    return subtract_inplace(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator-=(const Matrix& other) noexcept {
    return subtract_inplace(other);
}

// --- Multiplication ---

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::mul_elements(const Number scalar) const noexcept {
    Matrix<_rows, _cols, Number> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(), 
                   [scalar](const Number val){ return val * scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
[[nodiscard]] Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::mul_elements(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, Number> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   result.data_.begin(), 
                   std::multiplies<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::mul_elements_inplace(const Number scalar) noexcept {
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const Number val){ return val * scalar; });
    return *this;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::mul_elements_inplace(const Matrix& other) noexcept {
    Matrix<_rows, _cols, Number> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   data_.begin(), 
                   std::multiplies<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
template<size_type _other_cols>
Matrix<_rows, _other_cols, Number> Matrix<_rows, _cols, Number>::mul_matrix(const Matrix<_cols, _other_cols, Number>& other) const noexcept {
    Matrix<_rows, _other_cols, Number> result(0);
    // worth noticing that when compiling with -O2 or -O3 flags modern compilers will cache loop invariant results automatically
    for (size_type i = 0; i < _rows; ++i) {
        size_type i_mul_other_cols = i * _other_cols; // cache loop invariant result

        for (size_type k = 0; k < _cols; ++k) {
            Number a_ik = data_[i * _cols + k]; // cache the current result
            size_type k_mul_other_cols = k * _other_cols; // cache loop invariant result

            for (size_type j = 0; j < _other_cols; ++j) {
                result.data_[i_mul_other_cols + j] += a_ik * other.data_[k_mul_other_cols + j];
            }
        }
    }

    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
template<size_type _other_cols>
Matrix<_rows, _other_cols, Number> Matrix<_rows, _cols, Number>::operator%(const Matrix<_cols, _other_cols, Number>& other) const noexcept {
    return mul_matrix(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator*(const Number scalar) const noexcept {
    return mul_elements(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator*(const Matrix& other) const noexcept {
    return mul_elements(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator*=(const Number scalar) noexcept {
    return mul_elements_inplace(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator*=(const Matrix& other) noexcept {
    return mul_elements_inplace(other);
}

// --- Division ---

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::divide(const Number scalar) const noexcept {
    
    Number reciprocal = 0; 
    if constexpr (!std::numeric_limits<Number>::is_iec559) 
        assert(scalar != 0 && "FATAL: Matrix integer division by zero!");
    else 
        reciprocal = static_cast<Number>(1) / scalar;
    
    Matrix<_rows, _cols, Number> result;
    
    std::transform(std::execution::par_unseq, 
                   data_.begin(), data_.end(), result.data_.begin(), 
                   [scalar, reciprocal](Number val) {
                        if constexpr (std::numeric_limits<Number>::is_iec559) {
                            return val * reciprocal; // In case IEEE754 compliable type
                        } else {
                            return val / scalar;     // Other (integer) types
                        }
                   });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::divide_inplace(const Number scalar) noexcept {
    Number reciprocal = 0; 
    if constexpr (!std::numeric_limits<Number>::is_iec559) 
        assert(scalar != 0 && "FATAL: Matrix integer division by zero!");
    else 
        reciprocal = static_cast<Number>(1) / scalar;

    std::transform(std::execution::par_unseq, 
                   data_.begin(), data_.end(), data_.begin(), 
                   [scalar, reciprocal](Number val) {
                        if constexpr (std::numeric_limits<Number>::is_iec559) {
                            return val * reciprocal; // In case IEEE754 compliable type
                        } else {
                            return val / scalar;     // Other (integer) types
                        }
                   });
    return *this;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number> Matrix<_rows, _cols, Number>::operator/(const Number scalar) const noexcept {
    return divide(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, Number>& Matrix<_rows, _cols, Number>::operator/=(const Number scalar) noexcept {
    return divide_inplace(scalar);
}

// --- Logical Operators ---

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::negate() const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    std::negate<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator!() const noexcept {
    return negate();
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::equal_to<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::equal(const Number scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const Number val) { return val == scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::not_equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::not_equal_to<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::not_equal(const Number scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const Number val) { return val != scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::smaller(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::less<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::smaller(const Number scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const Number val) { return val < scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::smaller_equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::less_equal<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::smaller_equal(const Number scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const Number val) { return val <= scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::greater(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::greater<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::greater(const Number scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const Number val) { return val > scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::greater_equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::greater_equal<>());
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::greater_equal(const Number scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const Number val) { return val >= scalar; });
    return result;
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator==(const Matrix& other) const noexcept {
    return equal(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator==(const Number scalar) const noexcept {
    return equal(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator!=(const Matrix& other) const noexcept {
    return not_equal(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator!=(const Number scalar) const noexcept {
    return not_equal(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator<(const Matrix& other) const noexcept {
    return smaller(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator<(const Number scalar) const noexcept {
    return smaller(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator<=(const Matrix& other) const noexcept {
    return smaller_equal(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator<=(const Number scalar) const noexcept {
    return smaller_equal(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator>(const Matrix& other) const noexcept {
    return greater(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator>(const Number scalar) const noexcept {
    return greater(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator>=(const Matrix& other) const noexcept {
    return greater_equal(other);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, Number>::operator>=(const Number scalar) const noexcept {
    return greater_equal(scalar);
}

template<size_type _rows, size_type _cols, NumberLike Number>
Number Matrix<_rows, _cols, Number>::operator[](size_type row, size_type col) const noexcept {
    return data_[row * _cols + col];
}

template<size_type _rows, size_type _cols, NumberLike Number>
Number& Matrix<_rows, _cols, Number>::operator[](size_type row, size_type col) noexcept {
    return data_[row * _cols + col];
}

} // namespace linalg