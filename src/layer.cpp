#include "../include/layer.hpp"
#include <cmath>

Layer::Layer(int in_size, int out_size, std::unique_ptr<Activation> act) : input_size(in_size), output_size(out_size), activation(std::move(act)) {
    initialize_weights();
}

void Layer::initialize_weights() {    
    float stddev = std::sqrt(1.0f / input_size);
        
    weights = af::randn(output_size, input_size, f32) * stddev;
        
    biases = af::constant(0.0f, output_size, 1, f32);  
}

af::array Layer::forward(const af::array& input) {    
    input_cache = input;
    
    // Z = W * X + B    
    int current_batch_size = input_cache.dims(1);    
    linear_cache = af::matmul(weights, input_cache) + af::tile(biases, 1, current_batch_size);

        
    output_cache = activation->forward(linear_cache);
    
    return output_cache;
}

af::array Layer::backward(const af::array& grad_output, float learning_rate) {    
    af::array grad_linear = grad_output * activation->backward(linear_cache);
            
    af::array grad_weights = af::matmul(grad_linear, af::transpose(input_cache));
            
    af::array grad_biases = af::sum(grad_linear, 1);
            
    af::array grad_input = af::matmul(af::transpose(weights), grad_linear);
            
    update_weights(grad_weights, grad_biases, learning_rate);
    
    return grad_input;
}

void Layer::update_weights(const af::array& grad_weights, const af::array& grad_biases, float learning_rate) {    
    weights -= learning_rate * grad_weights;
    biases -= learning_rate * grad_biases;
}
