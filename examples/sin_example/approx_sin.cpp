#include "../../include/network.hpp"
#include "../../include/activ.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <string>
#include <algorithm>

/**
 * @brief Train a network on the sine function with adaptive learning rate.
 * 
 * @param net Network to train.
 * @param epochs Number of training epochs.
 * @param initial_lr Initial learning rate.
 * @param inputs Training inputs (x values).
 * @param targets Training targets (sin(x) values).
 * @return Final average loss.
 */
double train_sine(Network& net, int epochs, double initial_lr, const std::vector<double>& inputs, const std::vector<double>& targets) {
    double lr = initial_lr;
    double prev_loss = 1e9;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Normalize input to [-1, 1] since sine outputs in [-1, 1]
            Matrix input(1, 1);
            input(0, 0) = inputs[i] / (2.0 * M_PI);
            
            Matrix target(1, 1);
            target(0, 0) = targets[i];

            total_loss += net.train(input, target, lr);
        }

        double avg_loss = total_loss / inputs.size();

        // Adaptive learning rate 
        if (epoch > 0 && epoch % 500 == 0) {
            double adapt = prev_loss / avg_loss;
            if (avg_loss >= prev_loss) {
                // Loss increased: decrease learning rate
                lr *= std::min(adapt, 0.5);
            } else {
                // Loss decreased: increase learning rate
                lr *= std::clamp(adapt, 1.1, 2.0);
            }
            prev_loss = avg_loss;
        }

        // Log progress every 1000 epochs
        if ((epoch + 1) % 100 == 0) {
            std::cout << "  Epoch " << (epoch + 1) << "/" << epochs << ", Loss: " << std::scientific << avg_loss << ", LR: " << std::fixed << lr << std::endl;
        }
    }

    // Compute final loss
    double final_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Matrix input(1, 1);
        input(0, 0) = inputs[i] / (2.0 * M_PI);
        Matrix output = net.forward(input);
        double err = output(0, 0) - targets[i];
        final_loss += err * err;
    }
    return final_loss / inputs.size();
}

/**
 * @brief Save predictions to a CSV.
 * 
 * @param net Trained network.
 * @param filename Output filename.
 */
void save_predictions(Network& net, const std::string& filename = "../output/sine_predictions.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    file << "x,true_sin,predicted_sin,error\n";

    // Generate predictions for x in [-2pi, 2pi]
    const int num_points = 500;
    for (int i = 0; i <= num_points; ++i) {
        double x = -2.0 * M_PI + (4.0 * M_PI * i) / num_points;
        double true_value = std::sin(x);

        Matrix input(1, 1);
        input(0, 0) = x / (2.0 * M_PI);
        
        Matrix output = net.predict(input);
        double predicted = output(0, 0);
        double error = std::abs(predicted - true_value);

        file << x << "," << true_value << "," << predicted << "," << error << "\n";
    }

    file.close();
    std::cout << "Predictions saved to " << filename << std::endl;
}

int main() {
    std::cout << "Started" << std::endl;
    std::srand(static_cast<unsigned>(std::time(nullptr)));               

    std::cout << "Creating network..." << std::endl;
    // 1. Create network activations    
    Network net;
    net.add_layer(std::make_unique<Layer>(1, 32, std::make_unique<GELU>()));
    net.add_layer(std::make_unique<Layer>(32, 32, std::make_unique<GELU>()));
    net.add_layer(std::make_unique<Layer>(32, 1, std::make_unique<Linear>()));

    std::cout << "Network created." << std::endl;    
    std::cout << "Generating training data..." << std::endl;

    // 2. Generate training data: y = sin(x), x in [-2pi, 2pi]
    std::vector<double> inputs, targets;
    for (double x = -2.0 * M_PI; x <= 2.0 * M_PI; x += 0.05) {
        inputs.push_back(x);
        targets.push_back(std::sin(x));
    }    

    std::cout << "Training data generated." << std::endl;

    // 3. Train the network with adaptive learning rate
    double initial_lr = 0.01;
    int epochs = 15000;
    
    std::cout << "Start training" << std::endl << std::endl;

    double final_mse = train_sine(net, epochs, initial_lr, inputs, targets);

    std::cout << std::endl << "Training complete" << std::endl;

    // 4. Save model
    net.save("../models/sine_model.txt");    
    // 5. Save predictions
    save_predictions(net, "../output/sine_predictions.csv");    

    return 0;
}