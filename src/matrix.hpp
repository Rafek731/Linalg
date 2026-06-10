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

/**
 * @class Matrix
 * @brief A fixed-size, compile-time matrix template with SIMD-optimized operations.
 * 
 * The Matrix class provides a compile-time fixed-size matrix container with optimized
 * arithmetic, logical, and comparison operations. All dimensions are known at compile time,
 * enabling aggressive compiler optimizations. The class uses std::array internally for
 * contiguous storage and supports both scalar and element-wise matrix operations.
 * 
 * @tparam _rows Number of rows (must be > 0)
 * @tparam _cols Number of columns (must be > 0)
 * @tparam NumberLike The numeric type stored in the matrix (default: double)
 * 
 * @note All operations are marked noexcept and use parallel execution policies where applicable.
 * @warning Matrix dimensions must be positive integers greater than zero.
 */
template <size_type _rows, size_type _cols, Number NumberLike = double>
class Matrix {
    static_assert(_rows > 0 && _cols > 0, "Matrix dimensions (_rows and _cols) must be greater than 0.");
public:

    // ============================================================================
    // Constructors and Factory Methods
    // ============================================================================

    /// @brief Default constructor. Initializes matrix with uninitialized values.
    Matrix() = default;
    
    /**
     * @brief Constructs a matrix filled with a specific value.
     * @param init_value The value to fill all matrix elements with
     */
    explicit Matrix(NumberLike init_value) : Matrix() { data_.fill(init_value); };

    // ============================================================================
    // Dimension Query Methods
    // ============================================================================

    /**
     * @brief Returns the number of rows in the matrix.
     * @return constexpr size_type The number of rows (_rows)
     */
    [[nodiscard]] static constexpr size_type rows() noexcept { return _rows; }
    
    /**
     * @brief Returns the number of columns in the matrix.
     * @return constexpr size_type The number of columns (_cols)
     */
    [[nodiscard]] static constexpr size_type cols() noexcept { return _cols; }
    
    /**
     * @brief Returns the total number of elements in the matrix.
     * @return constexpr size_type The total size (rows * cols)
     */
    [[nodiscard]] static constexpr size_type size() noexcept { return size_; }

    // ============================================================================
    // Arithmetic Operators - Addition
    // ============================================================================

    /**
     * @brief Adds a scalar value to each element.
     * @param scalar The scalar value to add
     * @return Matrix A new matrix with the scalar added to each element
     */
    [[nodiscard]] Matrix add (const NumberLike scalar)      const noexcept;
    
    /**
     * @brief Performs element-wise addition with another matrix.
     * @param other The matrix to add
     * @return Matrix A new matrix containing the element-wise sum
     */
    [[nodiscard]] Matrix add(const Matrix& other)           const noexcept;
    
    /**
     * @brief Addition operator: adds a scalar to each element.
     * @param scalar The scalar value to add
     * @return Matrix A new matrix with the scalar added to each element
     */
    [[nodiscard]] Matrix operator+(const NumberLike scalar) const noexcept;
    
    /**
     * @brief Addition operator: performs element-wise addition.
     * @param other The matrix to add
     * @return Matrix A new matrix containing the element-wise sum
     */
    [[nodiscard]] Matrix operator+(const Matrix& other)     const noexcept;
    
    /**
     * @brief Adds a scalar value to each element in-place.
     * @param scalar The scalar value to add
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& add_inplace(const NumberLike scalar) noexcept;
    
    /**
     * @brief Performs element-wise addition in-place.
     * @param other The matrix to add
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& add_inplace(const Matrix& other)     noexcept;
    
    /**
     * @brief In-place addition operator: adds a scalar to each element.
     * @param scalar The scalar value to add
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator+= (const NumberLike scalar) noexcept;
    
    /**
     * @brief In-place addition operator: performs element-wise addition.
     * @param other The matrix to add
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator+= (const Matrix& other)     noexcept;

    // ============================================================================
    // Arithmetic Operators - Subtraction
    // ============================================================================

    /**
     * @brief Subtracts a scalar value from each element.
     * @param scalar The scalar value to subtract
     * @return Matrix A new matrix with the scalar subtracted from each element
     */
    [[nodiscard]] Matrix subtract (const NumberLike scalar) const noexcept;
    
    /**
     * @brief Performs element-wise subtraction with another matrix.
     * @param other The matrix to subtract
     * @return Matrix A new matrix containing the element-wise difference
     */
    [[nodiscard]] Matrix subtract (const Matrix& other)     const noexcept;
    
    /**
     * @brief Subtraction operator: subtracts a scalar from each element.
     * @param scalar The scalar value to subtract
     * @return Matrix A new matrix with the scalar subtracted from each element
     */
    [[nodiscard]] Matrix operator-(const NumberLike scalar) const noexcept;
    
    /**
     * @brief Subtraction operator: performs element-wise subtraction.
     * @param other The matrix to subtract
     * @return Matrix A new matrix containing the element-wise difference
     */
    [[nodiscard]] Matrix operator-(const Matrix& other)     const noexcept;
    
    /**
     * @brief Subtracts a scalar value from each element in-place.
     * @param scalar The scalar value to subtract
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& subtract_inplace(const NumberLike scalar) noexcept;
    
    /**
     * @brief Performs element-wise subtraction in-place.
     * @param other The matrix to subtract
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& subtract_inplace(const Matrix& other)     noexcept;
    
    /**
     * @brief In-place subtraction operator: subtracts a scalar from each element.
     * @param scalar The scalar value to subtract
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator-=(const NumberLike scalar)       noexcept;
    
    /**
     * @brief In-place subtraction operator: performs element-wise subtraction.
     * @param other The matrix to subtract
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator-=(const Matrix& other)           noexcept;

    // ============================================================================
    // Arithmetic Operators - Multiplication
    // ============================================================================

    /**
     * @brief Multiplies each element by a scalar (element-wise scalar multiplication).
     * @param scalar The scalar value to multiply by
     * @return Matrix A new matrix with each element multiplied by the scalar
     */
    [[nodiscard]] Matrix mul_elements(const NumberLike scalar) const noexcept;
    
    /**
     * @brief Multiplies each element by a scalar in-place.
     * @param scalar The scalar value to multiply by
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& mul_elements_inplace(const NumberLike scalar) noexcept;
    
    /**
     * @brief Performs element-wise multiplication (Hadamard product).
     * @param other The matrix to multiply element-wise
     * @return Matrix A new matrix containing the element-wise product
     */
    [[nodiscard]] Matrix mul_elements(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise multiplication in-place (Hadamard product).
     * @param other The matrix to multiply element-wise
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& mul_elements_inplace(const Matrix& other) noexcept;

    /**
     * @brief Performs matrix multiplication (standard matrix product).
     * @tparam _other_cols Number of columns in the other matrix
     * @param other A matrix with dimensions (_cols x _other_cols)
     * @return Matrix<_rows, _other_cols, NumberLike> A new matrix with dimensions (_rows x _other_cols)
     * @note This operation has O(n³) complexity for square matrices.
     * @note The inner loops are cache-optimized for modern processors.
     */
    template<size_type _other_cols>
    [[nodiscard]] Matrix<_rows, _other_cols, NumberLike> mul_matrix(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept;
    
    /**
     * @brief Matrix multiplication operator (% for matrix product).
     * @tparam _other_cols Number of columns in the other matrix
     * @param other A matrix with dimensions (_cols x _other_cols)
     * @return Matrix<_rows, _other_cols, NumberLike> A new matrix with dimensions (_rows x _other_cols)
     */
    template<size_type _other_cols>
    [[nodiscard]] Matrix<_rows, _other_cols, NumberLike> operator%(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept;
    
    /**
     * @brief Scalar multiplication operator.
     * @param scalar The scalar value to multiply by
     * @return Matrix A new matrix with each element multiplied by the scalar
     */
    [[nodiscard]] Matrix operator*(const NumberLike scalar) const noexcept;
    
    /**
     * @brief In-place scalar multiplication operator.
     * @param scalar The scalar value to multiply by
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator*=(const NumberLike scalar) noexcept;
    
    /**
     * @brief Element-wise multiplication operator (Hadamard product).
     * @param other The matrix to multiply element-wise
     * @return Matrix A new matrix containing the element-wise product
     */
    [[nodiscard]] Matrix operator*(const Matrix& other) const noexcept;
    
    /**
     * @brief In-place element-wise multiplication operator (Hadamard product).
     * @param other The matrix to multiply element-wise
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator*=(const Matrix& other) noexcept;

    // ============================================================================
    // Arithmetic Operators - Division
    // ============================================================================

    /**
     * @brief Divides each element by a scalar.
     * @param scalar The scalar divisor
     * @return Matrix A new matrix with each element divided by the scalar
     * @warning For integer types, asserts that scalar is non-zero.
     *          For floating-point types, uses reciprocal multiplication for efficiency.
     */
    [[nodiscard]] Matrix divide (const NumberLike scalar)    const noexcept;
    
    /**
     * @brief Division operator: divides each element by a scalar.
     * @param scalar The scalar divisor
     * @return Matrix A new matrix with each element divided by the scalar
     */
    [[nodiscard]] Matrix operator/ (const NumberLike scalar) const noexcept;
    
    /**
     * @brief Divides each element by a scalar in-place.
     * @param scalar The scalar divisor
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& divide_inplace(const NumberLike scalar) noexcept;
    
    /**
     * @brief In-place division operator: divides each element by a scalar.
     * @param scalar The scalar divisor
     * @return Matrix& Reference to this matrix after modification
     */
    Matrix& operator/=(const NumberLike scalar)     noexcept;

    // ============================================================================
    // Logical Operators
    // ============================================================================

    /**
     * @brief Performs element-wise logical negation.
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is the logical negation of the corresponding element
     * @note Non-zero elements are considered true and negated to false; zero elements are considered false and negated to true.
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> negate() const noexcept;
    
    /**
     * @brief Logical negation operator.
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is the logical negation of the corresponding element
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!() const noexcept;

    // ============================================================================
    // Comparison Operators - Equality
    // ============================================================================

    /**
     * @brief Performs element-wise equality comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if the corresponding elements are equal
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> equal(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise equality comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> equal(const NumberLike scalar) const noexcept;

    /**
     * @brief Performs element-wise inequality comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if the corresponding elements are not equal
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> not_equal(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise inequality comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if not equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> not_equal(const NumberLike scalar) const noexcept;

    // ============================================================================
    // Comparison Operators - Ordering
    // ============================================================================

    /**
     * @brief Performs element-wise less-than comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if less than the corresponding element
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise less-than comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if less than the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller(const NumberLike scalar) const noexcept;

    /**
     * @brief Performs element-wise less-than-or-equal comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if less than or equal to the corresponding element
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller_equal(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise less-than-or-equal comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if less than or equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> smaller_equal(const NumberLike scalar) const noexcept;

    /**
     * @brief Performs element-wise greater-than comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if greater than the corresponding element
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> greater(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise greater-than comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if greater than the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> greater(const NumberLike scalar) const noexcept;

    /**
     * @brief Performs element-wise greater-than-or-equal comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if greater than or equal to the corresponding element
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> greater_equal(const Matrix& other) const noexcept;
    
    /**
     * @brief Performs element-wise greater-than-or-equal comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if greater than or equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> greater_equal(const NumberLike scalar) const noexcept;
    

    // ============================================================================
    // Comparison Operator Overloads
    // ============================================================================

    /**
     * @brief Equality operator: element-wise equality comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix of element-wise equality results
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator==(const Matrix& other) const noexcept;
    
    /**
     * @brief Equality operator: element-wise equality comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator==(const NumberLike scalar) const noexcept;

    /**
     * @brief Inequality operator: element-wise inequality comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix of element-wise inequality results
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!=(const Matrix& other) const noexcept;
    
    /**
     * @brief Inequality operator: element-wise inequality comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if not equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator!=(const NumberLike scalar) const noexcept;

    /**
     * @brief Less-than operator: element-wise less-than comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix of element-wise comparison results
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator< (const Matrix& other) const noexcept;
    
    /**
     * @brief Less-than operator: element-wise less-than comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if less than the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator< (const NumberLike scalar) const noexcept;

    /**
     * @brief Less-than-or-equal operator: element-wise comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix of element-wise comparison results
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator<=(const Matrix& other) const noexcept;
    
    /**
     * @brief Less-than-or-equal operator: element-wise comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if less than or equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator<=(const NumberLike scalar) const noexcept;

    /**
     * @brief Greater-than operator: element-wise greater-than comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix of element-wise comparison results
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator> (const Matrix& other) const noexcept;
    
    /**
     * @brief Greater-than operator: element-wise greater-than comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if greater than the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator> (const NumberLike scalar) const noexcept;

    /**
     * @brief Greater-than-or-equal operator: element-wise comparison with another matrix.
     * @param other The matrix to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix of element-wise comparison results
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator>=(const Matrix& other) const noexcept;
    
    /**
     * @brief Greater-than-or-equal operator: element-wise comparison with a scalar.
     * @param scalar The scalar value to compare with
     * @return Matrix<_rows, _cols, bool> A boolean matrix where each element is true if greater than or equal to the scalar
     */
    [[nodiscard]] Matrix<_rows, _cols, bool> operator>=(const NumberLike scalar) const noexcept;

    // ============================================================================
    // Indexing Operators
    // ============================================================================

    /**
     * @brief Mutable subscript operator for element access.
     * @param row The row index (0-based)
     * @param col The column index (0-based)
     * @return NumberLike& Reference to the element at (row, col)
     */
    [[nodiscard]] NumberLike& operator[](size_type row, size_type col) noexcept;
    
    /**
     * @brief Const subscript operator for element access.
     * @param row The row index (0-based)
     * @param col The column index (0-based)
     * @return NumberLike The value at (row, col)
     */
    [[nodiscard]] NumberLike  operator[](size_type row, size_type col) const noexcept;

    // ============================================================================
    // Iterator Definitions
    // ============================================================================

    // ============================================================================
    // Row Iterators
    // ============================================================================

    /**
     * @struct row_iterator
     * @brief Contiguous iterator for iterating over matrix elements row-by-row.
     * 
     * This iterator provides random-access iteration over matrix elements in row-major order.
     * It maintains a pointer to a matrix element and allows all standard random-access operations.
     */
    struct row_iterator {
        using iterator_category = std::contiguous_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NumberLike;
        using pointer           = NumberLike*;
        using reference         = NumberLike&;

        /**
         * @brief Constructs a row iterator from a pointer.
         * @param pointer Pointer to the element to iterate from
         */
        row_iterator(NumberLike* const pointer) : ptr_(pointer) 
        {}

        /**
         * @brief Dereference operator: accesses the current element.
         * @return reference Reference to the current element
         */
        [[nodiscard]] inline reference operator*() const noexcept { return *ptr_; }
        /**
         * @brief Member access operator.
         * @return pointer Pointer to the current element
         */
        [[nodiscard]] inline pointer operator->() const noexcept { return ptr_; }

        /**
         * @brief Pre-increment operator.
         * @return row_iterator& Reference to this iterator after incrementing
         */
        row_iterator& operator++() noexcept {
            ++ptr_;
            return *this; 
        }  
        
        /**
         * @brief Post-increment operator.
         * @return row_iterator A copy of this iterator before incrementing
         */
        row_iterator operator++(int) noexcept {
            row_iterator result = *this; 
            ++(*this); 
            return result; 
        }
        
        /**
         * @brief Pre-decrement operator.
         * @return row_iterator& Reference to this iterator after decrementing
         */
        row_iterator& operator--() noexcept {
            --ptr_;
            return *this; 
        }  

        /**
         * @brief Post-decrement operator.
         * @return row_iterator A copy of this iterator before decrementing
         */
        row_iterator operator--(int) noexcept {
            row_iterator result = *this; 
            --(*this); 
            return result; 
        }

        /**
         * @brief Calculates the distance between two iterators.
         * @param other The iterator to calculate distance to
         * @return difference_type The distance (in elements)
         */
        inline difference_type distance(const row_iterator& other) const noexcept { return ptr_ - other.ptr_; }

        /**
         * @brief Equality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if both iterators point to the same element
         */
        [[nodiscard]] inline friend bool operator==(const row_iterator a, const row_iterator b) noexcept { return a.ptr_ == b.ptr_; }
        
        /**
         * @brief Inequality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if iterators point to different elements
         */
        [[nodiscard]] inline friend bool operator!=(const row_iterator a, const row_iterator b) noexcept { return a.ptr_ != b.ptr_; }
        /**
         * @brief Addition operator: advances iterator by offset.
         * @param a The iterator
         * @param jump The number of elements to advance
         * @return row_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend row_iterator operator+(const row_iterator& a, difference_type jump) noexcept { return row_iterator(a.ptr_ + jump); }
        
        /**
         * @brief Addition operator (commutative): advances iterator by offset.
         * @param jump The number of elements to advance
         * @param a The iterator
         * @return row_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend row_iterator operator+(difference_type jump, const row_iterator& a) noexcept { return row_iterator(a.ptr_ + jump); }
        
        /**
         * @brief Subtraction operator: creates iterator offset backward.
         * @param a The iterator
         * @param jump The number of elements to move backward
         * @return row_iterator A new iterator at the decreased position
         */
        [[nodiscard]] inline friend row_iterator operator-(const row_iterator& a, difference_type jump) noexcept { return row_iterator(a.ptr_ - jump); }
        
        /**
         * @brief Distance calculation operator.
         * @param other The other iterator
         * @return difference_type The distance between iterators
         */
        [[nodiscard]] inline difference_type operator-(const row_iterator& other) const noexcept { return distance(other); }

        /**
         * @brief In-place addition operator.
         * @param jump The number of elements to advance
         * @return row_iterator& Reference to this iterator after advancing
         */
        row_iterator& operator+=(difference_type jump) noexcept { ptr_ += jump; return *this; }
        
        /**
         * @brief In-place subtraction operator.
         * @param jump The number of elements to move backward
         * @return row_iterator& Reference to this iterator after moving backward
         */
        row_iterator& operator-=(difference_type jump) noexcept { ptr_ -= jump; return *this; }
        
        /**
         * @brief Less-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier position than b
         */
        [[nodiscard]] inline friend bool operator< (const row_iterator& a, const row_iterator& b) { return a.ptr_ <  b.ptr_; }
        
        /**
         * @brief Less-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier or equal position than b
         */
        [[nodiscard]] inline friend bool operator<=(const row_iterator& a, const row_iterator& b) { return a.ptr_ <= b.ptr_; }
        
        /**
         * @brief Greater-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later position than b
         */
        [[nodiscard]] inline friend bool operator> (const row_iterator& a, const row_iterator& b) { return a.ptr_ >  b.ptr_; }
        
        /**
         * @brief Greater-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later or equal position than b
         */
        [[nodiscard]] inline friend bool operator>=(const row_iterator& a, const row_iterator& b) { return a.ptr_ >= b.ptr_; }

        /**
         * @brief Subscript operator for random access.
         * @param n The offset from the current position
         * @return reference Reference to the element at offset n
         */
        [[nodiscard]] inline reference operator[](difference_type n) const noexcept { return *(ptr_ + n); }

    private:
        pointer ptr_;
    };
    
    /**
     * @struct const_row_iterator
     * @brief Const contiguous iterator for iterating over matrix elements row-by-row.
     * 
     * This iterator provides const random-access iteration over matrix elements in row-major order.
     * It maintains a pointer to a const matrix element and allows all standard random-access operations.
     */
    struct const_row_iterator {
        using iterator_category = std::contiguous_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NumberLike;
        using pointer           = const NumberLike*;
        using reference         = const NumberLike&;

        /**
         * @brief Constructs a const row iterator from a pointer.
         * @param pointer Pointer to the element to iterate from
         */
        const_row_iterator(NumberLike* const pointer) : ptr_(pointer) 
        {}

        /**
         * @brief Dereference operator: accesses the current element.
         * @return reference Const reference to the current element
         */
        [[nodiscard]] inline reference operator*() const noexcept { return *ptr_; }
        /**
         * @brief Member access operator.
         * @return pointer Const pointer to the current element
         */
        [[nodiscard]] inline pointer operator->() const noexcept { return ptr_; }

        /**
         * @brief Pre-increment operator.
         * @return const_row_iterator& Reference to this iterator after incrementing
         */
        const_row_iterator& operator++() noexcept {
            ++ptr_;
            return *this; 
        }  

        /**
         * @brief Post-increment operator.
         * @return const_row_iterator A copy of this iterator before incrementing
         */
        const_row_iterator operator++(int) noexcept {
            const_row_iterator result = *this; 
            ++(*this); 
            return result; 
        }

        /**
         * @brief Pre-decrement operator.
         * @return const_row_iterator& Reference to this iterator after decrementing
         */
        const_row_iterator& operator--() noexcept {
            --ptr_;
            return *this; 
        }  

        /**
         * @brief Post-decrement operator.
         * @return const_row_iterator A copy of this iterator before decrementing
         */
        const_row_iterator operator--(int) noexcept {
            const_row_iterator result = *this; 
            --(*this); 
            return result; 
        }
        
        /**
         * @brief Calculates the distance between two iterators.
         * @param other The iterator to calculate distance to
         * @return difference_type The distance (in elements)
         */
        [[nodiscard]] inline difference_type distance(const const_row_iterator& other) const noexcept { return ptr_ - other.ptr_; }

        /**
         * @brief Equality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if both iterators point to the same element
         */
        [[nodiscard]] inline friend bool operator==(const const_row_iterator a, const const_row_iterator b) noexcept { return a.ptr_ == b.ptr_; }
        
        /**
         * @brief Inequality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if iterators point to different elements
         */
        [[nodiscard]] inline friend bool operator!=(const const_row_iterator a, const const_row_iterator b) noexcept { return a.ptr_ != b.ptr_; }
        /**
         * @brief Addition operator: advances iterator by offset.
         * @param a The iterator
         * @param jump The number of elements to advance
         * @return const_row_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend const_row_iterator operator+(const const_row_iterator& a, difference_type jump) noexcept { return const_row_iterator(a.ptr_ + jump); }
        
        /**
         * @brief Addition operator (commutative): advances iterator by offset.
         * @param jump The number of elements to advance
         * @param a The iterator
         * @return const_row_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend const_row_iterator operator+(difference_type jump, const const_row_iterator& a) noexcept { return const_row_iterator(a.ptr_ + jump); }
        
        /**
         * @brief Subtraction operator: creates iterator offset backward.
         * @param a The iterator
         * @param jump The number of elements to move backward
         * @return const_row_iterator A new iterator at the decreased position
         */
        [[nodiscard]] inline friend const_row_iterator operator-(const const_row_iterator& a, difference_type jump) noexcept { return const_row_iterator(a.ptr_ - jump); }
        
        /**
         * @brief Distance calculation operator.
         * @param other The other iterator
         * @return difference_type The distance between iterators
         */
        [[nodiscard]] inline difference_type operator-(const const_row_iterator& other) const noexcept { return distance(other); }

        /**
         * @brief In-place addition operator.
         * @param jump The number of elements to advance
         * @return const_row_iterator& Reference to this iterator after advancing
         */
        const_row_iterator& operator+=(difference_type jump) noexcept { ptr_ += jump; return *this; }
        
        /**
         * @brief In-place subtraction operator.
         * @param jump The number of elements to move backward
         * @return const_row_iterator& Reference to this iterator after moving backward
         */
        const_row_iterator& operator-=(difference_type jump) noexcept { ptr_ -= jump; return *this; }
        
        /**
         * @brief Less-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier position than b
         */
        [[nodiscard]] inline friend bool operator< (const const_row_iterator& a, const const_row_iterator& b) noexcept { return a.ptr_ <  b.ptr_; }
        
        /**
         * @brief Less-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier or equal position than b
         */
        [[nodiscard]] inline friend bool operator<=(const const_row_iterator& a, const const_row_iterator& b) noexcept { return a.ptr_ <= b.ptr_; }
        
        /**
         * @brief Greater-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later position than b
         */
        [[nodiscard]] inline friend bool operator> (const const_row_iterator& a, const const_row_iterator& b) noexcept { return a.ptr_ >  b.ptr_; }
        
        /**
         * @brief Greater-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later or equal position than b
         */
        [[nodiscard]] inline friend bool operator>=(const const_row_iterator& a, const const_row_iterator& b) noexcept { return a.ptr_ >= b.ptr_; }

        /**
         * @brief Subscript operator for random access.
         * @param n The offset from the current position
         * @return reference Const reference to the element at offset n
         */
        [[nodiscard]] inline reference operator[](difference_type n) const noexcept { return *(ptr_ + n); }
    private:
        pointer ptr_;
    };
    
    // ============================================================================
    // Column Iterators (Future Implementation)
    // ============================================================================

    /**
     * @struct column_iterator
     * @brief Column iterator for iterating over matrix elements column-by-column.
     * 
     * @note This iterator is not yet implemented.
     */
    struct column_iterator {
        /// @todo Implement column-wise iteration
    };

    /**
     * @struct const_column_iterator
     * @brief Const column iterator for iterating over matrix elements column-by-column.
     * 
     * @note This iterator is not yet implemented.
     */
    struct const_column_iterator {
        /// @todo Implement const column-wise iteration
    };

    // ============================================================================
    // Strided Iterators
    // ============================================================================

    /**
     * @struct strided_iterator
     * @brief Random-access iterator with a configurable stride for skipping elements.
     * 
     * This iterator allows iterating over non-contiguous elements by providing a stride value.
     * It's useful for iterating over matrix rows, columns, or diagonals efficiently.
     */
    struct strided_iterator {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NumberLike;
        using pointer           = NumberLike*;
        using reference         = NumberLike&;

        /**
         * @brief Constructs a strided iterator.
         * @param pointer Pointer to the starting element
         * @param stride The step size between elements (if 0, defaults to 1)
         */
        strided_iterator(NumberLike* const pointer, difference_type stride) : ptr_(pointer), stride_(stride) {
            if(stride_ == 0)
                stride_ = 1;
        }

        /**
         * @brief Dereference operator: accesses the current element.
         * @return reference Reference to the current element
         */
        [[nodiscard]] inline reference operator*() const noexcept { return *ptr_; }
        /**
         * @brief Member access operator.
         * @return pointer Pointer to the current element
         */
        [[nodiscard]] inline pointer operator->() const noexcept { return ptr_; }

        /**
         * @brief Pre-increment operator: advances by stride.
         * @return strided_iterator& Reference to this iterator after incrementing
         */
        strided_iterator& operator++() noexcept {
            ptr_ += stride_;
            return *this; 
        }  

        /**
         * @brief Post-increment operator: advances by stride.
         * @return strided_iterator A copy of this iterator before incrementing
         */
        strided_iterator operator++(int) noexcept {
            strided_iterator result = *this; 
            ++(*this);
            return result; 
        }
        
        /**
         * @brief Pre-decrement operator: moves backward by stride.
         * @return strided_iterator& Reference to this iterator after decrementing
         */
        strided_iterator& operator--() noexcept { ptr_ -= stride_; return *this; }
        
        /**
         * @brief Post-decrement operator: moves backward by stride.
         * @return strided_iterator A copy of this iterator before decrementing
         */
        strided_iterator operator--(int) noexcept {
            strided_iterator result = *this; 
            --(*this); 
            return result; 
        }

        /**
         * @brief Calculates the distance between two iterators (in strides).
         * @param other The iterator to calculate distance to
         * @return difference_type The distance in terms of iterations
         */
        [[nodiscard]] inline difference_type distance(const strided_iterator& other) const noexcept { return (ptr_ - other.ptr_) / stride_; }

        /**
         * @brief Equality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if both iterators point to the same element
         */
        [[nodiscard]] inline friend bool operator==(const strided_iterator a, const strided_iterator b) noexcept { return a.ptr_ == b.ptr_; }
        
        /**
         * @brief Inequality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if iterators point to different elements
         */
        [[nodiscard]] inline friend bool operator!=(const strided_iterator a, const strided_iterator b) noexcept { return a.ptr_ != b.ptr_; }
        /**
         * @brief Addition operator: advances iterator by offset strides.
         * @param a The iterator
         * @param jump The number of strides to advance
         * @return strided_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend strided_iterator operator+(const strided_iterator& a, difference_type jump)   noexcept { return strided_iterator(a.ptr_ + (jump * a.stride_), a.stride_); }
        
        /**
         * @brief Addition operator (commutative): advances iterator by offset strides.
         * @param jump The number of strides to advance
         * @param a The iterator
         * @return strided_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend strided_iterator operator+(difference_type jump, const strided_iterator& a)         noexcept { return strided_iterator(a.ptr_ + (jump * a.stride_), a.stride_); }
        
        /**
         * @brief Subtraction operator: creates iterator offset backward by strides.
         * @param a The iterator
         * @param jump The number of strides to move backward
         * @return strided_iterator A new iterator at the decreased position
         */
        [[nodiscard]] inline friend strided_iterator operator-(const strided_iterator& a, difference_type jump)   noexcept { return strided_iterator(a.ptr_ - (jump * a.stride_), a.stride_); }
        
        /**
         * @brief Distance calculation operator.
         * @param other The other iterator
         * @return difference_type The distance between iterators (in strides)
         */
        [[nodiscard]] inline difference_type operator-(const strided_iterator& other) const noexcept { return distance(other); }

        /**
         * @brief In-place addition operator.
         * @param jump The number of strides to advance
         * @return strided_iterator& Reference to this iterator after advancing
         */
        strided_iterator& operator+=(difference_type jump) noexcept { ptr_ += jump * stride_; return *this; }
        
        /**
         * @brief In-place subtraction operator.
         * @param jump The number of strides to move backward
         * @return strided_iterator& Reference to this iterator after moving backward
         */
        strided_iterator& operator-=(difference_type jump) noexcept { ptr_ -= jump * stride_; return *this; }
        
        /**
         * @brief Less-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier position than b
         */
        [[nodiscard]] inline friend bool operator< (const strided_iterator& a, const strided_iterator& b) noexcept { return a.ptr_ <  b.ptr_; }
        
        /**
         * @brief Less-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier or equal position than b
         */
        [[nodiscard]] inline friend bool operator<=(const strided_iterator& a, const strided_iterator& b) noexcept { return a.ptr_ <= b.ptr_; }
        
        /**
         * @brief Greater-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later position than b
         */
        [[nodiscard]] inline friend bool operator> (const strided_iterator& a, const strided_iterator& b) noexcept { return a.ptr_ >  b.ptr_; }
        
        /**
         * @brief Greater-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later or equal position than b
         */
        [[nodiscard]] inline friend bool operator>=(const strided_iterator& a, const strided_iterator& b) noexcept { return a.ptr_ >= b.ptr_; }

        /**
         * @brief Subscript operator for random access by strides.
         * @param n The offset (in strides) from the current position
         * @return reference Reference to the element at offset n (in strides)
         */
        [[nodiscard]] inline reference operator[](difference_type n) const noexcept { return *(ptr_ + (n * stride_)); }

    private:
        pointer ptr_;
        difference_type stride_;
    };

    /**
     * @struct const_strided_iterator
     * @brief Const random-access iterator with a configurable stride for skipping elements.
     * 
     * This iterator allows const iteration over non-contiguous elements by providing a stride value.
     * It's useful for const iteration over matrix rows, columns, or diagonals efficiently.
     */
    struct const_strided_iterator {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NumberLike;
        using pointer           = const NumberLike*;
        using reference         = const NumberLike&;

        /**
         * @brief Constructs a const strided iterator.
         * @param pointer Const pointer to the starting element
         * @param stride The step size between elements (if 0, defaults to 1)
         */
        const_strided_iterator(const NumberLike* const pointer, difference_type stride) : ptr_(pointer), stride_(stride) {
            if(stride_ == 0)
                stride_ = 1;
        }

        /**
         * @brief Dereference operator: accesses the current element.
         * @return reference Const reference to the current element
         */
        [[nodiscard]] inline reference operator*() const noexcept { return *ptr_; }
        /**
         * @brief Member access operator.
         * @return pointer Const pointer to the current element
         */
        [[nodiscard]] inline pointer operator->() const noexcept { return ptr_; }

        /**
         * @brief Pre-increment operator: advances by stride.
         * @return const_strided_iterator& Reference to this iterator after incrementing
         */
        const_strided_iterator& operator++() noexcept {
            ptr_ += stride_;
            return *this; 
        }  

        /**
         * @brief Post-increment operator: advances by stride.
         * @return const_strided_iterator A copy of this iterator before incrementing
         */
        const_strided_iterator operator++(int) noexcept {
            const_strided_iterator result = *this; 
            ++(*this);
            return result; 
        }

        /**
         * @brief Pre-decrement operator: moves backward by stride.
         * @return const_strided_iterator& Reference to this iterator after decrementing
         */
        const_strided_iterator& operator--() noexcept { ptr_ -= stride_; return *this; }
        
        /**
         * @brief Post-decrement operator: moves backward by stride.
         * @return const_strided_iterator A copy of this iterator before decrementing
         */
        const_strided_iterator operator--(int) noexcept {
            const_strided_iterator result = *this; 
            --(*this); 
            return result; 
        }
        
        /**
         * @brief Calculates the distance between two iterators (in strides).
         * @param other The iterator to calculate distance to
         * @return difference_type The distance in terms of iterations
         */
        [[nodiscard]] inline difference_type distance(const const_strided_iterator& other) const noexcept { return (ptr_ - other.ptr_) / stride_; }

        /**
         * @brief Equality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if both iterators point to the same element
         */
        [[nodiscard]] inline friend bool operator==(const const_strided_iterator a, const const_strided_iterator b) noexcept { return a.ptr_ == b.ptr_; }
        
        /**
         * @brief Inequality comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if iterators point to different elements
         */
        [[nodiscard]] inline friend bool operator!=(const const_strided_iterator a, const const_strided_iterator b) noexcept { return a.ptr_ != b.ptr_; }
        /**
         * @brief Addition operator: advances iterator by offset strides.
         * @param a The iterator
         * @param jump The number of strides to advance
         * @return const_strided_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend const_strided_iterator operator+(const const_strided_iterator& a, difference_type jump) noexcept { return const_strided_iterator(a.ptr_ + (jump * a.stride_), a.stride_); }
        
        /**
         * @brief Addition operator (commutative): advances iterator by offset strides.
         * @param jump The number of strides to advance
         * @param a The iterator
         * @return const_strided_iterator A new iterator at the advanced position
         */
        [[nodiscard]] inline friend const_strided_iterator operator+(difference_type jump, const const_strided_iterator& a) noexcept { return const_strided_iterator(a.ptr_ + (jump * a.stride_), a.stride_); }
        
        /**
         * @brief Subtraction operator: creates iterator offset backward by strides.
         * @param a The iterator
         * @param jump The number of strides to move backward
         * @return const_strided_iterator A new iterator at the decreased position
         */
        [[nodiscard]] inline friend const_strided_iterator operator-(const const_strided_iterator& a, difference_type jump) noexcept { return const_strided_iterator(a.ptr_ - (jump * a.stride_), a.stride_); }
        
        /**
         * @brief Distance calculation operator.
         * @param other The other iterator
         * @return difference_type The distance between iterators (in strides)
         */
        [[nodiscard]] inline difference_type operator-(const const_strided_iterator& other) const noexcept { return distance(other); }

        /**
         * @brief In-place addition operator.
         * @param jump The number of strides to advance
         * @return const_strided_iterator& Reference to this iterator after advancing
         */
        const_strided_iterator& operator+=(difference_type jump) noexcept { ptr_ += jump * stride_; return *this; }
        
        /**
         * @brief In-place subtraction operator.
         * @param jump The number of strides to move backward
         * @return const_strided_iterator& Reference to this iterator after moving backward
         */
        const_strided_iterator& operator-=(difference_type jump) noexcept { ptr_ -= jump * stride_; return *this; }
        
        /**
         * @brief Less-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier position than b
         */
        [[nodiscard]] inline friend bool operator< (const const_strided_iterator& a, const const_strided_iterator& b) noexcept { return a.ptr_ <  b.ptr_; }
        
        /**
         * @brief Less-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to an earlier or equal position than b
         */
        [[nodiscard]] inline friend bool operator<=(const const_strided_iterator& a, const const_strided_iterator& b) noexcept { return a.ptr_ <= b.ptr_; }
        
        /**
         * @brief Greater-than comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later position than b
         */
        [[nodiscard]] inline friend bool operator> (const const_strided_iterator& a, const const_strided_iterator& b) noexcept { return a.ptr_ >  b.ptr_; }
        
        /**
         * @brief Greater-than-or-equal comparison.
         * @param a First iterator
         * @param b Second iterator
         * @return bool True if a points to a later or equal position than b
         */
        [[nodiscard]] inline friend bool operator>=(const const_strided_iterator& a, const const_strided_iterator& b) noexcept { return a.ptr_ >= b.ptr_; }

        /**
         * @brief Subscript operator for random access by strides.
         * @param n The offset (in strides) from the current position
         * @return reference Const reference to the element at offset n (in strides)
         */
        [[nodiscard]] inline reference operator[](difference_type n) const noexcept { return *(ptr_ + (n * stride_)); }

    private:
        pointer ptr_;
        difference_type stride_;
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
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul_elements(const NumberLike scalar) const noexcept {
    Matrix<_rows, _cols, NumberLike> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(), 
                   result.data_.begin(), 
                   [scalar](const NumberLike val){ return val * scalar; });
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
[[nodiscard]] Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul_elements(const Matrix& other) const noexcept {
    Matrix<_rows, _cols, NumberLike> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   result.data_.begin(), 
                   std::multiplies<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::mul_elements_inplace(const NumberLike scalar) noexcept {
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   data_.begin(),
                   [scalar](const NumberLike val){ return val * scalar; });
    return *this;
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::mul_elements_inplace(const Matrix& other) noexcept {
    Matrix<_rows, _cols, NumberLike> result;
    std::transform(std::execution::unseq, 
                   data_.begin(), data_.end(),
                   other.data_.begin(),
                   data_.begin(), 
                   std::multiplies<>());
    return result;
}

template<size_type _rows, size_type _cols, Number NumberLike>
template<size_type _other_cols>
Matrix<_rows, _other_cols, NumberLike> Matrix<_rows, _cols, NumberLike>::mul_matrix(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept {
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
Matrix<_rows, _other_cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator%(const Matrix<_cols, _other_cols, NumberLike>& other) const noexcept {
    return mul_matrix(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator*(const NumberLike scalar) const noexcept {
    return mul_elements(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike> Matrix<_rows, _cols, NumberLike>::operator*(const Matrix& other) const noexcept {
    return mul_elements(other);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator*=(const NumberLike scalar) noexcept {
    return mul_elements_inplace(scalar);
}

template<size_type _rows, size_type _cols, Number NumberLike>
Matrix<_rows, _cols, NumberLike>& Matrix<_rows, _cols, NumberLike>::operator*=(const Matrix& other) noexcept {
    return mul_elements_inplace(other);
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
    return equal(scalar);
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