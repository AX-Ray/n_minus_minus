#pragma once

#include <vector>

class Matrix {
    public:
        int rows, cols;
        std::vector<float> data;
        
        Matrix(int r, int c);

        float& operator()(int r, int c);
        const float& operator()(int r, int c) const;

        Matrix operator*(const Matrix& other) const;        
};