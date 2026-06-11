#include <span>
#include "linalg.hpp"
#include "matrix.hpp"
#include "dynamic_matrix.hpp"

namespace linalg {

template <NumberLike Number>
class MatrixView {
    MatrixView(const Matrix& matrix, size_type start_row=0, size_type start_col=0, size_type rows=-1, size_type cols=-1);
    MatrixView(const DynamicMatrix& matrix, size_type start_row=0, size_type start_col=0, size_type rows=-1, size_type cols=-1);
    MatrixView(const MatrixView& other, size_type start_row=0, size_type start_col=0, size_type rows=-1, size_type cols=-1);
    MatrixView(MatrixView&& other) ;

    [[nodiscard]] size_type rows() noexcept { return rows_; }
    [[nodiscard]] size_type cols() noexcept { return cols_; }
    [[nodiscard]] size_type size() noexcept { return rows_ * cols_; }

    /// arithmetic operators
    [[nodiscard]] Matrix add (const Number scalar)       const noexcept;
    [[nodiscard]] Matrix add (const Matrix& other)           const noexcept;
    [[nodiscard]] Matrix operator+ (const Number scalar) const noexcept;
    [[nodiscard]] Matrix operator+ (const Matrix& other)     const noexcept;
    
    MatrixView& add_inplace(const Number scalar) noexcept;
    MatrixView& add_inplace(const Matrix& other)     noexcept;
    MatrixView& operator+= (const Number scalar) noexcept;
    MatrixView& operator+= (const Matrix& other)     noexcept;
    
    MatrixView& subtract_inplace(const Number scalar) noexcept;
    MatrixView& subtract_inplace(const Matrix& other)     noexcept;
    MatrixView& operator-=(const Number scalar)       noexcept;
    MatrixView& operator-=(const Matrix& other)           noexcept;
    
    MatrixView& mul_inplace(const Number scalar) noexcept;
    MatrixView& operator*=(const Number scalar) noexcept;

    MatrixView& divide_inplace(const Number scalar) noexcept;
    MatrixView& operator/=(const Number scalar)     noexcept;

    [[nodiscard]] DynamiMatrix<bool> negate() const ;
    [[nodiscard]] DynamiMatrix<bool> operator!() const ;

    [[nodiscard]] DynamiMatrix<bool> equal(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> equal(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> not_equal(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> not_equal(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> smaller(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> smaller(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> smaller_equal(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> smaller_equal(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> greater(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> greater(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> greater_equal(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> greater_equal(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> operator==(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> operator==(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> operator!=(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> operator!=(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> operator< (const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> operator< (const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> operator<=(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> operator<=(const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> operator> (const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> operator> (const Number scalar) const ;

    [[nodiscard]] DynamiMatrix<bool> operator>=(const Matrix& other) const ;
    [[nodiscard]] DynamiMatrix<bool> operator>=(const Number scalar) const ;

    [[nodiscard]] Number& operator[](size_type row, size_type col) ;
    [[nodiscard]] Number  operator[](size_type row, size_type col) const ;

    DynamicMatrix copy() const;
private:
    std::vector<std::span<Number>> view_;
    size_type start_row_{0}, start_col_{0}, rows_{0}, cols_{0};
}; // class MatrixView
} // namespace linalg