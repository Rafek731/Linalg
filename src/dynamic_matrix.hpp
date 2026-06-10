#include "matrix.hpp"

namespace linalg {

template<NumberLike Number>
class Matrix<linalg::dynamic, linalg::dynamic, Number> {

    Matrix();
    Matrix(size_type rows, size_type cols); 
    Matrix(Number init_value);
    Matrix(size_type rows, size_type cols, std::vector<Number> source);
    Matrix(size_type rows, size_type cols, std::vector<Number>&& source);
    Matrix(Matrix&& other) noexcept;
    
    size_type rows() const noexcept { return rows_; }
    size_type cols() const noexcept { return cols_; }
    size_type size() const noexcept { return data_.size(); }

    Matrix  add(const Number scalar) const;
    Matrix  add(const Matrix& other) const;
    Matrix& add_inplace(const Number scalar);
    Matrix& add_inplace(const Matrix& scalar);
    
    Matrix  subtract(const Number scalar) const;
    Matrix  subtract(const Matrix& other) const;
    Matrix& subtract_inplace(const Number scalar);
    Matrix& subtract_inplace(const Number scalar);

    Matrix  mul_elements(const Number scalar) const;
    Matrix  mul_elements(const Matrix& other) const;
    Matrix& mul_elements_inplace(const Number scalar);
    Matrix& mul_elements_inplace(const Matrix& other);

    Matrix  mul_matrix(const Matrix& other) const;
    Matrix& mul_matrix_inplace(const Matrix& other);
    
    Matrix  operator+ (const Number scalar) const;
    Matrix  operator+ (const Matrix& other) const;
    Matrix& operator+=(const Number scalar);
    Matrix& operator+=(const Matrix& scalar);
    
    Matrix  operator- (const Number scalar) const;
    Matrix  operator- (const Matrix& other) const;
    Matrix& operator-=(const Number scalar);
    Matrix& operator-=(const Number scalar);

    Matrix  operator* (const Number scalar) const;
    Matrix  operator* (const Matrix& other) const;
    Matrix& operator*=(const Number scalar);
    Matrix& operator*=(const Matrix& other);

    Matrix  operator% (const Matrix& other) const;
    Matrix& operator%=(const Matrix& other);

    Matrix& operator=(Matrix&& other);
    
    Number  operator[](size_type row, size_type column) const;
    Number& operator[](size_type row, size_type column);

    std::span<Number>       get_row(size_type row_idx);
    std::span<const Number> get_row(size_type row_idx) const;

    std::vector<Number>     get_col(size_type col_idx) const;
    std::vecotor<Number*>   get_col(size_type col_idx);

    std::span<Number> flatten() const;

    void transpose();
    void reshape(size_type new_rows, size_type new_cols);

    Matrix copy() const { return Matrix(*this); };

private:
    std::vector<Number> data_;
    size_type rows_{0}, cols_{0};
}; // class Matrix

} // namespace linalg