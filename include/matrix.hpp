#pragma once

#include <vector>

class Matrix {
    public:
        int rows, cols;
        std::vector<double> data;
        
        Matrix(int r, int c);

        double& operator()(int r, int c);
        const double& operator()(int r, int c) const;

        Matrix operator*(const Matrix& other) const;
};