#include "matrix.hpp"

namespace linalg {
    template<NumberLike Number>
    class Matrix<linalg::dynamic, linalg::dynamic, Number> {

        Matrix();
        size_type 
        Matrix(Number init);
        Matrix();


    private:
        std::vector<Number> data_;
    }
}