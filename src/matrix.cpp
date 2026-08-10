#include "../include/matrix.hpp"

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(af::constant(0.0f, r, c, f32)) {}

Matrix::Matrix(int r, int c, af::array af_data) : rows(r), cols(c), data(af_data) {}

Matrix Matrix::operator*(const Matrix& other) const {    
    af::array result_data = af::matmul(this->data, other.data);    
    return Matrix(rows, other.cols, result_data);
}

float Matrix::operator()(int r, int c) const {    
    return data(r, c).scalar<float>();
}
