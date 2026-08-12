#pragma once

#include <arrayfire.h>

class Matrix {
private:
    int rows;
    int cols;
    af::array data; 

public:
    Matrix(int r, int c);    
    Matrix(int r, int c, af::array af_data); 
    
    float operator()(int r, int c) const; 
    
    Matrix operator*(const Matrix& other) const;
};
