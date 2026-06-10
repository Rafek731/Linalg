#include <gtest/gtest.h>
#include <numeric>
#include <iterator>
#include <algorithm>
#include "../src/matrix.hpp"

using namespace linalg;

// ============================================================================
// 1. INITIALIZATION & STATE TESTS
// ============================================================================

TEST(MatrixTest, BasicStateAndDimensions) {
    Matrix<3, 4, int> m(7);

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);

    auto val_0_0 = m[0, 0];
    auto val_1_2 = m[1, 2];
    auto val_2_3 = m[2, 3];

    EXPECT_EQ(val_0_0, 7);
    EXPECT_EQ(val_1_2, 7);
    EXPECT_EQ(val_2_3, 7);
}

// ============================================================================
// 2. ARITHMETIC OPERATORS (SCALAR & MATRIX)
// ============================================================================

TEST(MatrixArithmeticTest, ScalarAdditionAndSubtraction) {
    Matrix<2, 2, double> m(5.0);
    
    auto added = m + 2.5;
    auto add_0_0 = added[0, 0];
    EXPECT_DOUBLE_EQ(add_0_0, 7.5);

    auto subtracted = m - 1.5;
    auto sub_0_0 = subtracted[0, 0];
    EXPECT_DOUBLE_EQ(sub_0_0, 3.5);
}

TEST(MatrixArithmeticTest, MatrixElementWiseAdditionAndSubtraction) {
    Matrix<2, 2, int> A(10);
    Matrix<2, 2, int> B(3);

    auto C = A + B;
    auto val_C = C[0, 0];
    EXPECT_EQ(val_C, 13);

    auto D = A - B;
    auto val_D = D[0, 0];
    EXPECT_EQ(val_D, 7);
}

TEST(MatrixArithmeticTest, ScalarMultiplication) {
    Matrix<2, 2, float> m(4.0f);
    auto result = m * 2.5f;

    auto res_0_0 = result[0, 0];
    EXPECT_FLOAT_EQ(res_0_0, 10.0f);
}

// ============================================================================
// 3. IN-PLACE OPERATIONS & METHOD CHAINING
// ============================================================================

TEST(MatrixArithmeticTest, InPlaceChainingAndMutation) {
    Matrix<2, 2, double> m(2.0);

    // Complex sequence via method chaining: ((M + 3.0) * 2.0) - 1.0
    // Initial: 2.0 -> Add 3.0 (5.0) -> Mul 2.0 (10.0) -> Sub 1.0 (9.0)
    m.add_inplace(3.0).mul_inplace(2.0).subtract_inplace(1.0);

    auto val_0_0 = m[0, 0];
    auto val_1_1 = m[1, 1];

    EXPECT_DOUBLE_EQ(val_0_0, 9.0);
    EXPECT_DOUBLE_EQ(val_1_1, 9.0);

    // Verify operator+= overloads mutate correctly
    m += 1.0; 
    auto updated_val = m[0, 0];
    EXPECT_DOUBLE_EQ(updated_val, 10.0);
}

// ============================================================================
// 4. ADVANCED MATRIX MULTIPLICATION (NON-SQUARE MECHANICS)
// ============================================================================

TEST(MatrixArithmeticTest, MatrixMultiplicationNonSquare) {
    // A: 2 x 3 Matrix
    Matrix<2, 3, double> A;
    A[0, 0] = 1.0; A[0, 1] = 2.0; A[0, 2] = 3.0;
    A[1, 0] = 4.0; A[1, 1] = 5.0; A[1, 2] = 6.0;

    // B: 3 x 2 Matrix
    Matrix<3, 2, double> B;
    B[0, 0] = 7.0;  B[0, 1] = 8.0;
    B[1, 0] = 9.0;  B[1, 1] = 10.0;
    B[2, 0] = 11.0; B[2, 1] = 12.0;

    // Expected Result: 2 x 2 Matrix via the optimized i, k, j loop
    Matrix<2, 2, double> C = A * B;

    auto c00 = C[0, 0]; auto c01 = C[0, 1];
    auto c10 = C[1, 0]; auto c11 = C[1, 1];

    EXPECT_DOUBLE_EQ(c00, 58.0);
    EXPECT_DOUBLE_EQ(c01, 64.0);
    EXPECT_DOUBLE_EQ(c10, 139.0);
    EXPECT_DOUBLE_EQ(c11, 154.0);
}

// ============================================================================
// 5. DIVISION LOGIC (IEEE 754 RECIPROCALS vs INTEGRAL RUNTIMES)
// ============================================================================

TEST(MatrixDivisionTest, FloatingPointDivisionReciprocal) {
    Matrix<2, 2, double> m(10.0);
    auto result = m / 2.0; 
    
    auto res_0_0 = result[0, 0];
    EXPECT_DOUBLE_EQ(res_0_0, 5.0);
}

TEST(MatrixDivisionTest, IntegerDivisionTruncation) {
    Matrix<2, 2, int> m(10);
    auto result = m / 3; // Integer branches truncate
    
    auto res_0_0 = result[0, 0];
    EXPECT_EQ(res_0_0, 3);
}

TEST(MatrixDivisionTest, IntegerDivisionByZeroCrashes) {
    Matrix<2, 2, int> m(10);
    
    // Asserts that your compile-time compile branches catch integer zero errors via assert()
    EXPECT_DEATH({ 
        [[maybe_unused]] auto result = m / 0; 
    }, "FATAL: Matrix integer division by zero!");
}

// ============================================================================
// 6. LOGICAL MASKS & COMPARISON EVALUATION
// ============================================================================

TEST(MatrixLogicalTest, NegationAndBooleanMasks) {
    Matrix<2, 2, int> A;
    A[0, 0] = 10; A[0, 1] = 20;
    A[1, 0] = 30; A[1, 1] = 40;

    // Scalar Comparison Evaluation
    Matrix<2, 2, bool> bool_mask = A > 25;

    auto b00 = bool_mask[0, 0]; // 10 > 25 (false)
    auto b01 = bool_mask[0, 1]; // 20 > 25 (false)
    auto b10 = bool_mask[1, 0]; // 30 > 25 (true)
    auto b11 = bool_mask[1, 1]; // 40 > 25 (true)

    EXPECT_FALSE(b00);
    EXPECT_FALSE(b01);
    EXPECT_TRUE(b10);
    EXPECT_TRUE(b11);

    // Matrix-to-Matrix Complete Element-wise Matching
    Matrix<2, 2, int> B;
    B[0, 0] = 10; B[0, 1] = 99; 
    B[1, 0] = 99; B[1, 1] = 40; 

    Matrix<2, 2, bool> eq_mask = (A == B);
    
    EXPECT_TRUE((eq_mask[0, 0]));
    EXPECT_FALSE((eq_mask[0, 1]));
    EXPECT_FALSE((eq_mask[1, 0]));
    EXPECT_TRUE((eq_mask[1, 1]));
}

// ============================================================================
// 7. ITERATOR CATEGORIES & CONCEPT COMPLIANCE
// ============================================================================

TEST(IteratorTraitTest, StandardLibraryConceptsVerification) {
    using Mat = Matrix<3, 3, double>;
    
    // Verifies row_iterator meets standard C++20 contiguous concepts 
    bool is_row_contiguous = std::contiguous_iterator<Mat::row_iterator>;
    bool is_const_row_contiguous = std::contiguous_iterator<Mat::const_row_iterator>;
    EXPECT_TRUE(is_row_contiguous);
    EXPECT_TRUE(is_const_row_contiguous);
    
    // Verifies strided layout meets random access layout constraints, but denies contiguous properties
    bool is_stride_random = std::random_access_iterator<Mat::strided_iterator>;
    bool is_stride_contiguous = std::contiguous_iterator<Mat::strided_iterator>;
    EXPECT_TRUE(is_stride_random);
    EXPECT_FALSE(is_stride_contiguous);
}

// ============================================================================
// 8. STRIDED ITERATOR CALCULATIONS & MATH TRAPS
// ============================================================================

TEST(IteratorMathTest, StridedPointerJumpsAndDistances) {
    Matrix<3, 3, int> m;
    // Sequential Grid Initialization
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    // [ 7, 8, 9 ]
    int val_tracker = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            m[i, j] = val_tracker++;
        }
    }

    // Isolate middle column (index 1), setting jump strides to 3
    // Elements targeted sequentially: 2, 5, 8
    auto* base_element_ptr = &m[0, 1];
    typename Matrix<3, 3, int>::strided_iterator it(base_element_ptr, 3);

    // Index verification and access checks
    EXPECT_EQ(*it, 2);
    EXPECT_EQ(it[1], 5);
    EXPECT_EQ(it[2], 8);

    // Operator+ jumps checking
    auto advanced_it = it + 2;
    EXPECT_EQ(*advanced_it, 8);

    // Signed distance correctness checks (Avoids unsigned casting loop variants underflow checks)
    auto positive_distance = advanced_it - it;
    auto negative_distance = it - advanced_it;
    EXPECT_EQ(positive_distance, 2); 
    EXPECT_EQ(negative_distance, -2); 

    // Mutation checks via assignment additions
    it += 1;
    EXPECT_EQ(*it, 5);

    // Inequality relational bounds tests
    EXPECT_TRUE(it < advanced_it);
    EXPECT_FALSE(it > advanced_it);
}

// ============================================================================
// 9. STANDARD ALGORITHM CONTEXT PACKS
// ============================================================================

TEST(IteratorIntegrationTest, StandardAlgorithmCompatibility) {
    Matrix<3, 3, int> m(10); 

    // Point raw references to row parameters
    typename Matrix<3, 3, int>::row_iterator start_row(&m[0, 0]);
    typename Matrix<3, 3, int>::row_iterator end_row(&m[0, 0] + 3);

    // Compute sum of row via <numeric> std::accumulate (10 + 10 + 10)
    int calculated_sum = std::accumulate(start_row, end_row, 0);
    EXPECT_EQ(calculated_sum, 30);

    // Validate container predicates via std::all_of
    bool matching_predicates = std::all_of(start_row, end_row, [](int value) { return value == 10; });
    EXPECT_TRUE(matching_predicates);
}