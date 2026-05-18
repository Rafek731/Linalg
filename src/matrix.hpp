#include <cstddef>
#include <array>
#include "number.hpp"


template <std::size_t rows, std::size_t cols, Number NumberLike = double>
class Matrix {
    static constexpr size_ = rows * cols;
    std::array<NumberLike, size_> data_;

public:

    Matrix() {}
    explicit Matrix(NumberLike __init_value) {
        for(auto &value: data_) 
            value = __init_value;
    }

    static constexpr std::size_t rows() const noexcept { return rows; }
    static constexpr std::size_t cols() const noexcept { return cols; }
    static constexpr std::size_t size() const noexcept { return size_; }

    /// arithmetic operators
    
    /// addition

    Matrix<rows, cols, Numberlike> operator+(const NumberLike __scalar) const {
        Matrix<rows, cols, NumberLike> tmp;
        for(std::size_t i = 0; i < size_; ++i)
            tmp.data_[i] = data_[i] + __scalar;
        return tmp;
    }

    Matrix<rows, cols, Numberlike> operator+(const Matrix<rows, cols, NumberLike>& other) const {
        Matrix<rows, cols, NumberLike> tmp;
        for(std::size_t i = 0; i < size_; ++i) 
            tmp.data_[i] = data_[i] + other.data_[i];
        return tmp;
    }

    Matrix<rows, cols, Numberlike>& operator+=(const NumberLike __scalar) noexcept {
        for(auto &value: data_) 
            value += __scalar;
        return *this;
    }

    Matrix<rows, cols, Numberlike>& operator+=(const Matrix<rows, cols, NumberLike>& other) noexcept {
        for(std::size_t i = 0; i < size_; ++i) 
            data_[i] += other.data_[i];
        return *this;
    }

    /// subtraction

    Matrix<rows, cols, Numberlike> operator-(const NumberLike __scalar) const {
        Matrix<rows, cols, NumberLike> tmp;
        for(std::size_t i = 0; i < size_; ++i)
            tmp.data_[i] = data_[i] - __scalar;
        return tmp;
    }

    Matrix<rows, cols, Numberlike> operator-(const Matrix<rows, cols, NumberLike>& other) const {
        Matrix<rows, cols, NumberLike> tmp;
        for(std::size_t i = 0; i < size_; ++i) 
            tmp.data_[i] = data_[i] - other.data_[i];
        return tmp;
    }
    
    Matrix<rows, cols, Numberlike>& operator-=(const NumberLike __scalar) noexcept {
        for(auto &value: data_) 
            value -= __scalar;
        return *this;
    }

    Matrix<rows, cols, Numberlike>& operator-=(const Matrix<rows, cols, NumberLike>& other) noexcept {
        for(std::size_t i = 0; i < size_; ++i) 
            data_[i] -= other.data_[i];
        return *this;
    }

    /// multiplication

    Matrix<rows, cols, Numberlike> operator*(const NumberLike __scalar) const {
        Matrix<rows, cols, NumberLike> tmp;
        for(std::size_t i = 0; i < size_; ++i) 
            tmp.data_[i] = data_[i] * other.data_[i];
        return tmp;
    }

    template<std::size_t other_rows, std::size_t other_cols>
    Matrix<rows, other_cols, Numberlike> operator*(const Matrix<cols, other_cols, NumberLike>& other) const {

    }

    Matrix<rows, cols, Numberlike>& operator*=(const NumberLike __scalar) noexcept {
        for(auto &value: data_) 
            value -= __scalar;
        return *this;
    }

    template<std::size_t other_rows, std::size_t other_cols>
    Matrix<rows, other_cols, Numberlike>& operator*=(const Matrix<cols, other_cols, NumberLike>& other) noexcept {
        return *this;
    }

    Matrix<rows, cols, Numberlike> operator/(const NumberLike __scalar) const {

    }

    Matrix<rows, cols, Numberlike>  operator/=(const NumberLike __scalar) noexcept {
        return *this;

    }


    /// logical operatrs

    Matrix<rows, cols, bool> operator==(const Matrix<rows, cols, NumberLike>& other) {

    } 

    Matrix<rows, cols, bool> operator!(const Matrix<rows, cols, NumberLike>& other) {

    }

    Matrix<rows, cols, bool> operator!=(const Matrix<rows, cols, NumberLike>& other) {

    }
    
    Matrix<rows, cols, bool> operator<(const Matrix<rows, cols, NumberLike>& other) {

    }

    Matrix<rows, cols, bool> operator<=(const Matrix<rows, cols, NumberLike>& other) {

    }

    Matrix<rows, cols, bool> operator>(const Matrix<rows, cols, NumberLike>& other) {
        
    }
    
    Matrix<rows, cols, bool> operator>=(const Matrix<rows, cols, NumberLike>& other) {

    }

    /// conversion operators
};