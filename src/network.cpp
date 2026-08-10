#include "../include/network.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>

void Network::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

af::array Network::forward(const af::array& input) {
    af::array current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

void Network::backward(const af::array& grad_output, float learning_rate) {
    af::array current_grad = grad_output;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        current_grad = layers[i]->backward(current_grad, learning_rate);
    }
}

float Network::train(const af::array& input, const af::array& target, float learning_rate) {        
    af::array output = forward(input);
                
    float loss = mse_loss(output, target);
            
    float batch_size = static_cast<float>(input.dims(1)); 
    af::array grad_output = (2.0f * (output - target)) / batch_size;
            
    backward(grad_output, learning_rate);
    
    return loss;
}

af::array Network::predict(const af::array& input) {
    return forward(input);
}

float Network::mse_loss(const af::array& output, const af::array& target) {    
    af::array diff = output - target;
        
    return af::mean<float>(diff * diff);
}

void Network::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot open" << std::endl;
        return;
    }
        
    file << layers.size() << std::endl;
        
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        file << layer->get_input_size() << " " << layer->get_output_size() << std::endl;
                        
        af::array weights = layer->get_weights();
        int w_rows = weights.dims(0);
        int w_cols = weights.dims(1);
        std::vector<float> h_weights(weights.elements());
        weights.host(h_weights.data()); 
                        
        for (int r = 0; r < w_rows; ++r) {
            for (int c = 0; c < w_cols; ++c) {                
                file << h_weights[c * w_rows + r] << " ";
            }
            file << std::endl;
        }
                        
        af::array biases = layer->get_biases();
        std::vector<float> h_biases(biases.elements());
        biases.host(h_biases.data());
                
        for (size_t b = 0; b < h_biases.size(); ++b) {
            file << h_biases[b] << " ";
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void Network::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot open" << std::endl;
        return;
    }
        
    size_t num_layers;
    file >> num_layers;
        
    if (num_layers != layers.size()) {
        std::cerr << "mismatch" << std::endl;
        file.close();
        return;
    }
        
    for (size_t i = 0; i < layers.size(); ++i) {
        int in_size, out_size;
        file >> in_size >> out_size;
                
        if (in_size != layers[i]->get_input_size() || 
            out_size != layers[i]->get_output_size()) {
            std::cerr << "mismatch" << std::endl;
            file.close();
            return;
        }
                        
        std::vector<float> h_weights(out_size * in_size);
        for (int r = 0; r < out_size; ++r) {
            for (int c = 0; c < in_size; ++c) {                
                file >> h_weights[c * out_size + r];
            }
        }        
        af::array weights(out_size, in_size, h_weights.data());
                
        std::vector<float> h_biases(out_size);
        for (int r = 0; r < out_size; ++r) {
            file >> h_biases[r];
        }
        af::array biases(out_size, 1, h_biases.data());
                
        layers[i]->set_weights(weights);
        layers[i]->set_biases(biases);
    }
    
    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
}