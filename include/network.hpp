#pragma once

#include <vector>
#include <memory>
#include <string>
#include <arrayfire.h> 
#include "layer.hpp"

class Network {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        
    public:
        Network() = default;
                
        void add_layer(std::unique_ptr<Layer> layer);
                        
        af::array forward(const af::array& input);
                        
        void backward(const af::array& grad_output, float learning_rate);
                        
        float train(const af::array& input, const af::array& target, float learning_rate);
                
        af::array predict(const af::array& input);
                
        float mse_loss(const af::array& output, const af::array& target);
                
        size_t get_layer_count() const { return layers.size(); }   
        
        void save(const std::string& filename) const;
        void load(const std::string& filename);
};
