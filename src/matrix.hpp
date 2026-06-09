#pragma once

#include <cstddef>
#include <array>
#include <algorithm> // std::transform
#include <execution> // std::excecution::unseq
#include <functional> // std::plus, std::minus, std::less, std::less_equal, std::greate, std::greater_equal, std::equal_to, std::not_equal_to
#include <iterator>

#include "linalg.hpp"

namespace linalg {
using size_type = std::size_t;

static constexpr size_type dynamic = static_cast<size_type>(-1); 

template <size_type _rows, size_type _cols, Number NumberLike = double>
class Matrix {
    static_assert(_rows > 0 && _cols > 0, "Matrix dimensions (_rows and _cols) must be greater than 0.");
public:

    Matrix() = default;
    explicit Matrix(NumberLike init_value) : Matrix() { data_.fill(init_value); };

    [[nodiscard]] static constexpr size_type rows() noexcept { return _rows; }
    [[nodiscard]] static constexpr size_type cols() noexcept { return _cols; }
    [[nodiscard]] static constexpr size_type size() noexcept { return size_; }

    /// arithmetic operators
    [[nodiscard]] Matrix add (const NumberLike scalar)       const noexcept;
    [[nodiscard]] Matrix add (const Matrix& other)           const noexcept;
    [[nodiscard]] Matrix operator+ (const NumberLike scalar) const noexcept;
    [[nodiscard]] Matrix operator+ (const Matrix& other)     const noexcept;
    
    Matrix& add_inplace(const NumberLike scalar) noexcept;
    Matrix& add_inplace(const Matrix& other)     noexcept;
    Matrix& operator+= (const NumberLike scalar) noexcept;
    Matrix& operator+= (const Matrix& other)     noexcept;

    [[nodiscard]] Matrix subtract (const NumberLike scalar) const noexcept;
    [[nodiscard]] Matrix subtract (const Matrix& other)     const noexcept;
    [[nodiscard]] Matrix operator-(const NumberLike scalar) const noexcept;
    [[nodiscard]] Matrix operator-(const Matrix& other)     const noexcept;
    
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

    [[nodiscard]] Matrix<_rows, _cols, bool> negate() const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!() const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> equal(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> not_equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> not_equal(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> smaller(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> smaller_equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller_equal(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> greater(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> greater(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> greater_equal(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> greater_equal(const NumberLike scalar) const noexcept;
    

    [[nodiscard]] Matrix<_rows, _cols, bool> operator==(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator==(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator!=(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!=(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator< (const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator< (const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator<=(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator<=(const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator> (const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator> (const NumberLike scalar) const noexcept;

    [[nodiscard]] Matrix<_rows, _cols, bool> operator>=(const Matrix& other) const noexcept;
    [[nodiscard]] Matrix<_rows, _cols, bool> operator>=(const NumberLike scalar) const noexcept;

    [[nodiscard]] NumberLike& operator[](size_type row, size_type col) noexcept;
    [[nodiscard]] NumberLike  operator[](size_type row, size_type col) const noexcept;
    
    struct flat_iterator {
        using iterator_category = std::contiguous_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NumberLike;
        using pointer           = NumberLike*;
        using reference         = NumberLike&;

        flat_iterator(NumberLike* const pointer) : ptr_(pointer) 
        {}

        inline reference operator*() const noexcept { return *ptr_; }
        inline pointer operator->() const noexcept { return ptr_; }

        flat_iterator& operator++() noexcept {
            ++ptr_;
            return *this; 
        }  

        flat_iterator operator++(int) noexcept {
            flat_iterator result = *this; 
            ++(*this); 
            return result; 
        }

        inline difference_type distance(const flat_iterator& other) const noexcept { return ptr_ - other.ptr_; }

        inline friend bool operator==(const flat_iterator a, const flat_iterator b) noexcept { return a.ptr_ == b.ptr_; }
        inline friend bool operator!=(const flat_iterator a, const flat_iterator b) noexcept { return a.ptr_ != b.ptr_; }
        inline friend flat_iterator operator+(const flat_iterator& a, size_type jump) noexcept { return flat_iterator(a.ptr_ + jump, a.stride_); }
        inline friend flat_iterator operator+(size_type jump, const flat_iterator& a) noexcept { return flat_iterator(a.ptr_ + jump, a.stride_); }
        inline friend flat_iterator operator-(const flat_iterator& a, size_type jump) noexcept { return flat_iterator(a.ptr_ - jump, a.stride_); }
        inline difference_type operator-(const flat_iterator& other) const noexcept { return distance(other); }

        
        flat_iterator& operator+=(size_type jump) noexcept { ptr_ += jump; return *this; }
        flat_iterator& operator-=(size_type jump) noexcept { ptr_ -= jump; return *this; }
        
        inline friend bool operator< (const flat_iterator& a, const flat_iterator& b) { return a.ptr_ < b.ptr; }
        inline friend bool operator<=(const flat_iterator& a, const flat_iterator& b) { return a.ptr_ <= b.ptr; }
        inline friend bool operator> (const flat_iterator& a, const flat_iterator& b) { return a.ptr_ > b.ptr; }
        inline friend bool operator>=(const flat_iterator& a, const flat_iterator& b) { return a.ptr_ >= b.ptr; }

    private:
        pointer ptr_;
    };
    
    struct const_flat_iterator {
        using iterator_category = std::contiguous_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NumberLike;
        using pointer           = const NumberLike*;
        using reference         = const NumberLike&;

        const_flat_iterator(NumberLike* const pointer) : ptr_(pointer) 
        {}

        inline reference operator*() const noexcept { return *ptr_; }
        inline pointer operator->() const noexcept { return ptr_; }

        const_flat_iterator& operator++() noexcept {
            ++ptr_;
            return *this; 
        }  

        const_flat_iterator operator++(int) noexcept {
            const_flat_iterator result = *this; 
            ++(*this); 
            return result; 
        }
        
        inline difference_type distance(const const_flat_iterator& other) const noexcept { return ptr_ - other.ptr_; }

        inline friend bool operator==(const const_flat_iterator a, const const_flat_iterator b) noexcept { return a.ptr_ == b.ptr_; }
        inline friend bool operator!=(const const_flat_iterator a, const const_flat_iterator b) noexcept { return a.ptr_ != b.ptr_; }
        inline friend const_flat_iterator operator+(const const_flat_iterator& a, size_type jump) noexcept { return const_flat_iterator(a.ptr_ + jump, a.stride_); }
        inline friend const_flat_iterator operator+(size_type jump, const const_flat_iterator& a) noexcept { return const_flat_iterator(a.ptr_ + jump, a.stride_); }
        inline friend const_flat_iterator operator-(const const_flat_iterator& a, size_type jump) noexcept { return const_flat_iterator(a.ptr_ - jump, a.stride_); }
        inline difference_type operator-(const const_flat_iterator& other) const noexcept { return distance(other); }

        
        const_flat_iterator& operator+=(size_type jump) noexcept { ptr_ += jump; return *this; }
        const_flat_iterator& operator-=(size_type jump) noexcept { ptr_ -= jump; return *this; }
        
        inline friend bool operator< (const const_flat_iterator& a, const const_flat_iterator& b) noexcept { return a.ptr_ < b.ptr; }
        inline friend bool operator<=(const const_flat_iterator& a, const const_flat_iterator& b) noexcept { return a.ptr_ <= b.ptr; }
        inline friend bool operator> (const const_flat_iterator& a, const const_flat_iterator& b) noexcept { return a.ptr_ > b.ptr; }
        inline friend bool operator>=(const const_flat_iterator& a, const const_flat_iterator& b) noexcept { return a.ptr_ >= b.ptr; }

    private:
        pointer ptr_;
    };
    
    struct flat_stride_iterator {
        
    };

    struct const_flat_stride_iterator {

    };

    struct flat_column_iterator {
        
    };

    struct flat_const_column_iterator {

    };
    
    struct stride_column_iterator {
        
    };

    struct stride_const_column_iterator {

    };

    struct diagonal_iterator {

    };

    struct const_diagonal_iterator {

    };


private:
    static constexpr size_type size_ = _rows * _cols;
    std::array<NumberLike, size_> data_;

    // Allows operators to access data_ across different sizes/types
    template <size_type _r, size_type _c, Number _N>
    friend class Matrix;
};

// --- Addition ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::add(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, NumberLike> result;
    
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(),
                   [scalar](const NumberLike val){ return val + scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::add(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, NumberLike> result;

    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(), 
                   result.data_.begin(), 
                   std::plus<>());

    return result;
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
    Matrix<_rows, _cols, NumberLike> result;
    
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(),
                   [scalar](const NumberLike val){ return val - scalar; });
    
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::subtract(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, NumberLike> result;

    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(), 
                   result.data_.begin(), 
                   std::minus<>());
    
    return result;
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
    Matrix<_rows, _cols, NumberLike> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(), 
                   [scalar](const NumberLike val){ return val * scalar; });
    return result;
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
    Matrix<_rows, _other_cols, NumberLike> result(0);
    // worth noticing that when compiling with -O2 or -O3 flags modern compilers will cache loop invariant results automatically
    for (size_type i = 0; i < _rows; ++i) {
        size_type i_mul_other_cols = i * _other_cols; // cache loop invariant result

        for (size_type k = 0; k < _cols; ++k) {
            NumberLike a_ik = data_[i * _cols + k]; // cache the current result
            size_type k_mul_other_cols = k * _other_cols; // cache loop invariant result

            for (size_type j = 0; j < _other_cols; ++j) {
                result.data_[i_mul_other_cols + j] += a_ik * other.data_[k_mul_other_cols + j];
            }
        }
    }

    return result;
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
    
    NumberLike reciprocal = 0; 
    if constexpr (!std::numeric_limits<NumberLike>::is_iec559) 
        assert(scalar != 0 && "FATAL: Matrix integer division by zero!");
    else 
        reciprocal = static_cast<NumberLike>(1) / scalar;
    
    Matrix<_rows, _cols, NumberLike> result;
    
    std::transform(std::execution::par_unseq, 
                   data_.begin(), data_.end(), result.data_.begin(), 
                   [scalar, reciprocal](NumberLike val) {
                        if constexpr (std::numeric_limits<NumberLike>::is_iec559) {
                            return val * reciprocal; // In case IEEE754 compliable type
                        } else {
                            return val / scalar;     // Other (integer) types
                        }
                   });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::divide_inplace(const NumberLike scalar) noexcept {
    NumberLike reciprocal = 0; 
    if constexpr (!std::numeric_limits<NumberLike>::is_iec559) 
        assert(scalar != 0 && "FATAL: Matrix integer division by zero!");
    else 
        reciprocal = static_cast<NumberLike>(1) / scalar;

    std::transform(std::execution::par_unseq, 
                   data_.begin(), data_.end(), data_.begin(), 
                   [scalar, reciprocal](NumberLike val) {
                        if constexpr (std::numeric_limits<NumberLike>::is_iec559) {
                            return val * reciprocal; // In case IEEE754 compliable type
                        } else {
                            return val / scalar;     // Other (integer) types
                        }
                   });
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator/(const NumberLike scalar) const noexcept {
    return divide(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator/=(const NumberLike scalar) noexcept {
    return divide_inplace(scalar);
}

// --- Logical Operators ---

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::negate() const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    std::negate<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!() const noexcept {
    return negate();
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::equal_to<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::equal(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const NumberLike val) { return val == scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::not_equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::not_equal_to<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::not_equal(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const NumberLike val) { return val != scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::smaller(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::less<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::smaller(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const NumberLike val) { return val < scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::smaller_equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::less_equal<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::smaller_equal(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const NumberLike val) { return val <= scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::greater(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::greater<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::greater(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const NumberLike val) { return val > scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::greater_equal(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    other.data_.begin(),
                    result.data_.begin(),
                    std::greater_equal<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::greater_equal(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, bool> result;
    std::transform (std::execution::unseq,
                    data_.begin(),
                    data_.end(),
                    result.data_.begin(),
                    [scalar](const NumberLike val) { return val >= scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator==(const Matrix& other) const noexcept {
    return equal(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator==(const NumberLike scalar) const noexcept {
    return equal(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!=(const Matrix& other) const noexcept {
    return not_equal(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator!=(const NumberLike scalar) const noexcept {
    return not_equal(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<(const Matrix& other) const noexcept {
    return smaller(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<(const NumberLike scalar) const noexcept {
    return smaller(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<=(const Matrix& other) const noexcept {
    return smaller_equal(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator<=(const NumberLike scalar) const noexcept {
    return smaller_equal(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>(const Matrix& other) const noexcept {
    return greater(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>(const NumberLike scalar) const noexcept {
    return greater(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>=(const Matrix& other) const noexcept {
    return greater_equal(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, bool> Matrix<_rows, _cols, NumberLike>::operator>=(const NumberLike scalar) const noexcept {
    return greater_equal(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
NumberLike Matrix<_rows, _cols, NumberLike>::operator[](size_type row, size_type col) const noexcept {
    return data_[row * _cols + col];
}

template<size_type _rows, size_type _cols, Number NumberLike>
NumberLike& Matrix<_rows, _cols, NumberLike>::operator[](size_type row, size_type col) noexcept {
    return data_[row * _cols + col];
}

} // namespace linalg