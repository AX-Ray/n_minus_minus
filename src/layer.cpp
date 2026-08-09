#include "../include/layer.hpp"
#include <random>

Layer::Layer(int in_size, int out_size, std::unique_ptr<Activation> act)
    : input_size(in_size)
    , output_size(out_size)
    , weights(out_size, in_size)   
    , biases(out_size, 1)          
    , activation(std::move(act))
    , input_cache(in_size, 1)
    , linear_cache(out_size, 1)
    , output_cache(out_size, 1)
{
    initialize_weights();
}

void Layer::initialize_weights() {    
    std::random_device rd;
    std::mt19937 gen(rd());
        
    double stddev = std::sqrt(1.0 / input_size);
    std::normal_distribution<double> dist(0.0, stddev);
    
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights(i, j) = dist(gen);
        }
        biases(i, 0) = 0.0;  
    }
}

Matrix Layer::forward(const Matrix& input) {    
    input_cache = input;
    
    // z = W * x + b    
    linear_cache = weights * input;
    for (int i = 0; i < output_size; ++i) {
        linear_cache(i, 0) += biases(i, 0);
    }
        
    output_cache = Matrix(output_size, 1);
    for (int i = 0; i < output_size; ++i) {
        output_cache(i, 0) = activation->forward(linear_cache(i, 0));
    }
    
    return output_cache;
}

Matrix Layer::backward(const Matrix& grad_output, double learning_rate) {    
    Matrix grad_linear(output_size, 1);
    for (int i = 0; i < output_size; ++i) {
        double da_dz = activation->backward(output_cache(i, 0));
        grad_linear(i, 0) = grad_output(i, 0) * da_dz;
    }
        
    Matrix grad_weights(output_size, input_size);
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            grad_weights(i, j) = grad_linear(i, 0) * input_cache(j, 0);
        }
    }
        
    Matrix grad_biases(output_size, 1);
    for (int i = 0; i < output_size; ++i) {
        grad_biases(i, 0) = grad_linear(i, 0);
    }
        
    Matrix grad_input(input_size, 1);
    for (int i = 0; i < input_size; ++i) {
        double sum = 0.0;
        for (int j = 0; j < output_size; ++j) {
            sum += weights(j, i) * grad_linear(j, 0);
        }
        grad_input(i, 0) = sum;
    }
        
    update_weights(grad_weights, grad_biases, learning_rate);
    
    return grad_input;
}

void Layer::update_weights(const Matrix& grad_weights, const Matrix& grad_biases, double learning_rate) {    
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights(i, j) -= learning_rate * grad_weights(i, j);
        }
        biases(i, 0) -= learning_rate * grad_biases(i, 0);
    }
}