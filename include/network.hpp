#pragma once

#include <vector>
#include <memory>
#include "layer.hpp"
#include "matrix.hpp"

class Network {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        
    public:
        Network() = default;
                
        void add_layer(std::unique_ptr<Layer> layer);
                
        Matrix forward(const Matrix& input);
                
        void backward(const Matrix& grad_output, double learning_rate);
                
        double train(const Matrix& input, const Matrix& target, double learning_rate);        
                
        Matrix predict(const Matrix& input);
                
        double mse_loss(const Matrix& output, const Matrix& target);
                
        size_t get_layer_count() const { return layers.size(); }  
        
        void copy_weights_from(const Network& other);
        
        void save(const std::string& filename) const;
        void load(const std::string& filename);
};