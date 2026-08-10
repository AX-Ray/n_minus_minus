#pragma once

#include <vector>
#include <memory>
#include <arrayfire.h> 
#include "activ.hpp"

class Layer {
    private:
        int input_size;
        int output_size;
                        
        af::array weights;
        af::array biases;
                
        std::unique_ptr<Activation> activation;
                        
        af::array input_cache;     
        af::array linear_cache;     // z = W * x + b 
        af::array output_cache;     // a = activation(z) 
        
    public:        
        Layer(int input_size, int output_size, std::unique_ptr<Activation> activation);
                        
        af::array forward(const af::array& input);
                
        af::array backward(const af::array& grad_output, float learning_rate);
                
        int get_output_size() const { return output_size; }
        int get_input_size() const { return input_size; }
                
        void update_weights(const af::array& grad_weights, const af::array& grad_biases, float learning_rate);
                
        void initialize_weights();  
                
        const af::array& get_weights() const { return weights; }
        const af::array& get_biases() const { return biases; }
        void set_weights(const af::array& new_weights) { weights = new_weights; }
        void set_biases(const af::array& new_biases) { biases = new_biases; }
};
