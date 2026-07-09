#include "../include/network.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>

void Network::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Matrix Network::forward(const Matrix& input) {
    Matrix current = input;
    
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    
    return current;
}

void Network::backward(const Matrix& grad_output, double learning_rate) {
    Matrix current_grad = grad_output;
        
    for (int i = layers.size() - 1; i >= 0; --i) {
        current_grad = layers[i]->backward(current_grad, learning_rate);
    }
}

double Network::train(const Matrix& input, const Matrix& target, double learning_rate) {    
    Matrix output = forward(input);
        
    double loss = mse_loss(output, target);
        
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        grad_output(i, 0) = 2.0 * (output(i, 0) - target(i, 0));
    }
        
    backward(grad_output, learning_rate);
    
    return loss;
}

Matrix Network::predict(const Matrix& input) {
    return forward(input);
}

double Network::mse_loss(const Matrix& output, const Matrix& target) {
    double sum = 0.0;
    for (int i = 0; i < output.rows; ++i) {
        double diff = output(i, 0) - target(i, 0);
        sum += diff * diff;
    }
    return sum / output.rows;
}

void Network::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing!" << std::endl;
        return;
    }
        
    file << layers.size() << std::endl;
        
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
                
        file << layer->get_input_size() << " " << layer->get_output_size() << std::endl;
                
        const Matrix& weights = layer->get_weights();
        for (int r = 0; r < weights.rows; ++r) {
            for (int c = 0; c < weights.cols; ++c) {
                file << weights(r, c) << " ";
            }
            file << std::endl;
        }
                
        const Matrix& biases = layer->get_biases();
        for (int r = 0; r < biases.rows; ++r) {
            file << biases(r, 0) << " ";
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void Network::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for reading!" << std::endl;
        return;
    }
        
    size_t num_layers;
    file >> num_layers;
        
    if (num_layers != layers.size()) {
        std::cerr << "Error: Network architecture mismatch! Expected " 
                  << layers.size() << " layers, got " << num_layers << std::endl;
        file.close();
        return;
    }
        
    for (size_t i = 0; i < layers.size(); ++i) {
        int in_size, out_size;
        file >> in_size >> out_size;
                
        if (in_size != layers[i]->get_input_size() || 
            out_size != layers[i]->get_output_size()) {
            std::cerr << "Error: Layer " << i << " size mismatch!" << std::endl;
            file.close();
            return;
        }
                
        Matrix weights(out_size, in_size);
        Matrix biases(out_size, 1);
                
        for (int r = 0; r < weights.rows; ++r) {
            for (int c = 0; c < weights.cols; ++c) {
                file >> weights(r, c);
            }
        }
                
        for (int r = 0; r < biases.rows; ++r) {
            file >> biases(r, 0);
        }
                
        layers[i]->set_weights(weights);
        layers[i]->set_biases(biases);
    }
    
    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
}