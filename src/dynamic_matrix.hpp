#pragma once

#include "matrix.hpp"

namespace linalg {

template<NumberLike Number>
class DynamicMatrix {

    DynamicMatrix();
    DynamicMatrix(size_type rows, size_type cols); 
    DynamicMatrix(Number init_value);
    DynamicMatrix(size_type rows, size_type cols, std::vector<Number> source);
    DynamicMatrix(size_type rows, size_type cols, std::vector<Number>&& source);
    DynamicMatrix(DynamicMatrix&& other) noexcept;
    
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type size() const noexcept { return data_.size(); }

    DynamicMatrix  add(const Number  scalar) const;
    DynamicMatrix  add(const DynamicMatrix& other) const;
    DynamicMatrix& add_inplace(const Number scalar);
    DynamicMatrix& add_inplace(const DynamicMatrix& other);
    
    DynamicMatrix  subtract(const Number  scalar) const;
    DynamicMatrix  subtract(const DynamicMatrix& other) const;
    DynamicMatrix& subtract_inplace(const Number scalar);
    DynamicMatrix& subtract_inplace(const DynamicMatrix& other);

    DynamicMatrix  mul_elements(const Number scalar) const;
    DynamicMatrix  mul_elements(const DynamicMatrix& other) const;
    DynamicMatrix& mul_elements_inplace(const Number scalar);
    DynamicMatrix& mul_elements_inplace(const DynamicMatrix& other);

    DynamicMatrix  mul_matrix(const DynamicMatrix& other) const;
    DynamicMatrix& mul_matrix_inplace(const DynamicMatrix& other);

    DynamicMatrix<bool> negate(const Number scalar);
    
    DynamicMatrix<bool> equal(const Number scalar);
    DynamicMatrix<bool> not_equal(const Number scalar);
    DynamicMatrix<bool> greater(const Number scalar);
    DynamicMatrix<bool> greater_equal(const Number scalar);
    DynamicMatrix<bool> smaller(const Number scalar);
    DynamicMatrix<bool> smaller_equal(const Number scalar);
    
    DynamicMatrix  operator+ (const Number  scalar) const;
    DynamicMatrix  operator+ (const DynamicMatrix& other) const;
    DynamicMatrix& operator+=(const Number  scalar);
    DynamicMatrix& operator+=(const DynamicMatrix& scalar);
    
    DynamicMatrix  operator- (const Number  scalar) const;
    DynamicMatrix  operator- (const DynamicMatrix& other) const;
    DynamicMatrix& operator-=(const Number  scalar);
    DynamicMatrix& operator-=(const Number  scalar);

    DynamicMatrix  operator* (const Number  scalar) const;
    DynamicMatrix  operator* (const DynamicMatrix& other) const;
    DynamicMatrix& operator*=(const Number  scalar);
    DynamicMatrix& operator*=(const DynamicMatrix& other);
    
    DynamicMatrix  operator% (const DynamicMatrix& other) const;
    DynamicMatrix& operator%=(const DynamicMatrix& other);
    
    DynamicMatrix<bool> operator!(const Number scalar);
    
    DynamicMatrix<bool> operator==(const Number scalar);
    DynamicMatrix<bool> operator!=(const Number scalar);
    DynamicMatrix<bool> operator> (const Number scalar);
    DynamicMatrix<bool> operator>=(const Number scalar);
    DynamicMatrix<bool> operator< (const Number scalar);
    DynamicMatrix<bool> operator<=(const Number scalar);

    DynamicMatrix& operator=(Matrix&& other);
    
    Number  operator[](size_type row, size_type column) const;
    Number& operator[](size_type row, size_type column);
    
    std::span<Number>       get_row(size_type row_idx);
    std::span<const Number> get_row(size_type row_idx) const;
    
    std::vector<Number>     get_col(size_type col_idx) const;
    std::vector<std::reference_wrapper<Number>> get_col(size_type col_idx);
    
    std::span<Number> flatten() const;
    
    void transpose();
    void reshape(size_type new_rows, size_type new_cols);
    
    DynamicMatrix copy() const { return DynamicMatrix(*this); };
    DynamicMatrix&& move();
    
    Matrix get_submatrix(size_type start_row, size_type start_col, size_type end_row, size_type end_col);
private:
    std::vector<Number> data_;
    size_type rows_{0}, cols_{0};
    static DynamicMatrix strassen_multiplication(const DynamicMatrix& A, const DynamicMatrix& B);
    void strassen_partition(const DynamicMatrix& A, DynamicMatrix& A11, DynamicMatrix& A12, DynamicMatrix& A21, DynamicMatrix& A22);
    DynamicMatrix strassen_join(DynamicMatrix& A11, DynamicMatrix& A12, DynamicMatrix& A21, DynamicMatrix& A22);
}; // class Matrix

template<NumberLike Number>
DynamicMatrix<Number> DynamicMatrix<Number>::strassen_multiplication(const DynamicMatrix& A, const DynamicMatrix& B) {
    if(A.rows_ < 2) 
        return DynamicMatrix(1, 1, A[0, 0] * B[0, 0]);
    
    // strassen_partition
    const size_type A_half_rows = A.rows_ >> 1;
    const size_type A_half_cols = A.cols_ >> 1;

    const size_type B_half_rows = B.rows_ >> 1;
    const size_type B_half_cols = B.cols_ >> 1;

    DynamicMatrix   A11(A_half_rows, A_half_cols),
                    A12(A_half_rows, A_half_cols),
                    A21(A_half_rows, A_half_cols),
                    A22(A_half_rows, A_half_cols),
                    B11(B_half_rows, B_half_cols),
                    B12(B_half_rows, B_half_cols),
                    B21(B_half_rows, B_half_cols),
                    B22(B_half_rows, B_half_cols);

    std::thread part_A(strassen_partition, A, A11, A12, A21, A22);
    std::thread part_B(strassen_partition, B, B11, B12, B21, B22);

    part_A.join();
    part_B.join();

    DynamicMatrix M1 = strassen_multiplication(A11 + A22,  B11 + B22);
    DynamicMatrix M2 = strassen_multiplication(A21 + A22,  B11      );
    DynamicMatrix M3 = strassen_multiplication(A11,        B12 - B22);
    DynamicMatrix M4 = strassen_multiplication(A22,        B21 - B11);
    DynamicMatrix M5 = strassen_multiplication(A11 + A12,  B22      );
    DynamicMatrix M6 = strassen_multiplication(A21 - A11,  B11 + B12);
    DynamicMatrix M7 = strassen_multiplication(A12 - A22,  B21 + B22);

    return strassen_join(M1 + M4 - M5 + M7, M3 + M5, M2 + M5, M1 - M2 + M3 + M6);
}

template<NumberLike Number>
void DynamicMatrix<Number>::strassen_partition(const DynamicMatrix& A, DynamicMatrix& A11, DynamicMatrix& A12, DynamicMatrix& A21, DynamicMatrix& A22) {
    const size_type half_rows = A.rows_ >> 1;
    const size_type half_cols = A.cols_ >> 1;

    for (size_type i = 0; i < half_rows; ++i) {
        for (size_type j = 0; j < half_cols; ++j) {
            A11[i, j] = A[i, j];
        }
    }
    
    for (size_type i = 0; i < half_rows; ++i) {
        for (size_type j = half_cols; j < A.cols_; ++j) {
            A12[i, j - half_cols] = A[i, j];
        }
    }

    for (size_type i = half_rows; i < A.rows_; ++i) {
        for (size_type j = 0; j < half_cols; ++j) {
            A11[i - half_rows, j] = A[i, j];
        }
    }
    
    for (size_type i = half_rows; i < A.rows_; ++i) {
        for (size_type j = half_cols; j < A.cols_; ++j) {
            A11[i - half_rows, j - half_cols] = A[i, j];
        }
    }
}


} // namespace linalg