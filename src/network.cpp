#include "../include/network.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>


namespace {
    std::unique_ptr<Activation> create_activation(const std::string& name) {
        if (name == "Sigmoid") return std::make_unique<Sigmoid>();
        if (name == "ReLU") return std::make_unique<ReLU>();
        if (name == "LeakyReLU") return std::make_unique<LeakyReLU>();
        if (name == "Linear") return std::make_unique<Linear>();
        if (name == "TanH") return std::make_unique<TanH>();
        if (name == "SELU") return std::make_unique<SELU>();
        if (name == "Swish") return std::make_unique<Swish>();
        if (name == "Mish") return std::make_unique<Mish>();
        if (name == "LSLU") return std::make_unique<LSLU>();
        if (name == "GELU") return std::make_unique<GELU>();
        if (name == "ELU") return std::make_unique<ELU>();
        if (name == "Softplus") return std::make_unique<Softplus>();
        if (name == "APLU") return std::make_unique<APLU>();
        throw std::runtime_error("Unknown activation name: " + name);
    }
}



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

void Network::copy_weights_from(const Network& other) {
    if (layers.size() != other.layers.size()) {
        std::cerr << "Layer count mismatch\n";
        return;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->set_weights(other.layers[i]->get_weights());
        layers[i]->set_biases(other.layers[i]->get_biases());
    }
}



void Network::save(std::ostream& os) const {
    os << layers.size() << std::endl;

    for (const auto& layer : layers) {
        os << layer->get_input_size() << " " << layer->get_output_size() << std::endl;
        os << layer->get_activation_name() << std::endl;

        const Matrix& weights = layer->get_weights();
        for (int r = 0; r < weights.rows; ++r) {
            for (int c = 0; c < weights.cols; ++c) {
                os << weights(r, c) << " ";
            }
            os << std::endl;
        }

        const Matrix& biases = layer->get_biases();
        for (int r = 0; r < biases.rows; ++r) {
            os << biases(r, 0) << " ";
        }
        os << std::endl;
    }
}

void Network::load(std::istream& is) {
    size_t num_layers;
    is >> num_layers;
    if (!is) {
        throw std::runtime_error("Failed to read number of layers");
    }

    layers.clear();

    for (size_t i = 0; i < num_layers; ++i) {
        int in, out;
        std::string act_name;
        is >> in >> out;
        is >> act_name;
        if (!is) {
            throw std::runtime_error("Failed to read layer metadata");
        }

        Matrix weights(out, in);
        for (int r = 0; r < weights.rows; ++r) {
            for (int c = 0; c < weights.cols; ++c) {
                is >> weights(r, c);
            }
        }

        Matrix biases(out, 1);
        for (int r = 0; r < biases.rows; ++r) {
            is >> biases(r, 0);
        }

        if (!is) {
            throw std::runtime_error("Failed to read weights/biases");
        }

        auto activation = create_activation(act_name);
        auto layer = std::make_unique<Layer>(in, out, std::move(activation));
        layer->set_weights(weights);
        layer->set_biases(biases);
        layers.push_back(std::move(layer));
    }
}

void Network::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    save(file);
    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void Network::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    try {
        load(file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        file.close();
        return;
    }
    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
}