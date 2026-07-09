#include "../include/matrix.hpp"

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

double& Matrix::operator()(int r, int c) {
    return data[r * cols + c];
}

const double& Matrix::operator()(int r, int c) const {
    return data[r * cols + c];
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);                        
            }
            result(i, j) = sum;
        }
    }
    return result;
}