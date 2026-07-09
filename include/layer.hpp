#pragma once

#include <vector>
#include <memory>
#include "matrix.hpp"
#include "activ.hpp"

class Layer {
    private:
        int input_size;
        int output_size;
                
        Matrix weights;
        Matrix biases;
                
        std::unique_ptr<Activation> activation;
                
        Matrix input_cache;     
        Matrix linear_cache;     // z = W * x + b 
        Matrix output_cache;     // a = activation(z) 
        
    public:        
        Layer(int input_size, int output_size, std::unique_ptr<Activation> activation);
                
        Matrix forward(const Matrix& input);
                
        Matrix backward(const Matrix& grad_output, double learning_rate);
                
        int get_output_size() const { return output_size; }
        int get_input_size() const { return input_size; }
                
        void update_weights(const Matrix& grad_weights, const Matrix& grad_biases, double learning_rate);
                
        void initialize_weights();  
        
        const Matrix& get_weights() const { return weights; }
        const Matrix& get_biases() const { return biases; }
        void set_weights(const Matrix& new_weights) { weights = new_weights; }
        void set_biases(const Matrix& new_biases) { biases = new_biases; }
};