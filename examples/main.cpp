#include "../include/network.hpp"
#include "../include/activ.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib> 
#include <ctime> 
#include <algorithm>


/**
 * @brief Train the neural network on quadratic function data.
 * 
 * Uses adaptive learning rate:
 * - If loss increases → decrease LR (multiply by 0.85 max)
 * - If loss decreases → increase LR (multiply by 1.1-1.35)
 * 
 * @param net Reference to the network to train
 * @param epochs Number of training epochs
 * @param l_r Initial learning rate
 * @param inputs Vector of input values (x)
 * @param targets Vector of target values (y = x^2)
 */
void net_train(Network& net, int epochs, double l_r, const std::vector<double>& inputs, const std::vector<double>& targets) {
    double prev_total_loss = 999999.0; // Previous loss value for LR adaptation

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
                
        // Iterate over all training examples
        for (int i = 0; i < inputs.size(); i++) {
            // Normalization: input in range [-0.3, 0.3] (x in [-30, 30] / 100)
            Matrix input(1, 1);
            input(0, 0) = inputs[i] / 100.0; 
            
            // Normalization: target in range [0, 9] (x² in [0, 900] / 10000)
            Matrix target(1, 1);
            target(0, 0) = targets[i] / 10000.0;  
            
            // Train on a single sample and accumulate loss
            total_loss += net.train(input, target, l_r);
        }  
        
        // Average loss over the epoch
        double current_epoch_loss = total_loss / inputs.size();

        // Logging every 100 epochs
        if (epoch > 0 && epoch % 100 == 0) {   
                        
            std::ofstream file("../logs/log.txt", std::ios::app); 
            
            if (file.is_open()) {
                file << current_epoch_loss << " " << l_r <<  std::endl;
                file.close(); 
            } else {
                std::cerr << "Cannot open log.txt" << std::endl;
            }
        }
        
        // Adaptive learning rate adjustment every 500 epochs.
        if (epoch > 0 && epoch % 500 == 0) {    
            double adapt = prev_total_loss / current_epoch_loss;
            if (current_epoch_loss >= prev_total_loss) {                
                l_r *= std::min(adapt, 0.85);            
            } else {                      
                l_r *= std::clamp(adapt, 1.1, 1.35); 
            }
            prev_total_loss = current_epoch_loss; 
        }

        // Progress output every 1000 epochs
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", loss: " << current_epoch_loss << " (LR: " << l_r << ")" << std::endl;
        }
    }
}


/**
 * @brief Initialize the neural network architecture.
 * 
 * Structure: 1 - 20 - 20 - 1
 * 
 * - Input layer: 1 neuron (receives x)
 * 
 * - Hidden layer 1: 20 neurons with LSLU activation
 * 
 * - Hidden layer 2: 20 neurons with LSLU activation
 * 
 * - Output layer: 1 neuron with linear activation (returns prediction)
 * 
 * @param net Reference to the network to initialize
 */
void net_init(Network& net) { 
    net.add_layer(std::make_unique<Layer>(1, 20, std::make_unique<LSLU>()));       
    net.add_layer(std::make_unique<Layer>(20, 20, std::make_unique<LSLU>()));       
    net.add_layer(std::make_unique<Layer>(20, 1, std::make_unique<Linear>()));       
}

/**
 * @brief Save network predictions to a text file.
 * 
 * Generates predictions for x in [-100, 100] with step 0.5
 * and saves in format: "Input: x -> y_pred"
 * 
 * @param net Trained network
 * @param filename Output file name (default: "output/predict.txt")
 */
void predict_save(Network& net, const std::string& filename = "../output/predict.txt") {
    std::ofstream file(filename);
        
    for (double x = -100.0; x <= 100.0; x += 0.5) {
        Matrix input(1, 1);
        input(0, 0) = x / 100.0;  // Normalization matching training
        
        Matrix output = net.predict(input);        

        file << "Input: " << x << " -> " << output(0, 0) * 10000.0 << std::endl; // Denormalization
    }   
    
    file.close();
}


/**
 * @brief Main program entry point.
 * 
 * Steps:
 * 
 * 1. Initialize random number generator
 * 
 * 2. Clear log file
 * 
 * 3. Create and initialize the network
 * 
 * 4. Load saved model (if exists)
 * 
 * 5. Generate training data: y = x^2 on [-30, 30]
 * 
 * 6. Train the network
 * 
 * 7. Save predictions
 * 
 * 8. Save the model
 */
int main() {    
    std::srand(time(nullptr)); 

    std::ofstream clear_file("../logs/log.txt", std::ios::trunc);
    clear_file.close(); 

    Network net;        
    net_init(net);
    net.load("../models/model.txt");
        
    std::vector<double> inputs, targets;
    for (double x = -30.0; x <= 30.0; x += 0.5) {
        inputs.push_back(x);
        targets.push_back(x * x);
    } 
        
    double learning_rate = 0.2;  
    int epochs = 5000;         
        
    
    net_train(net, epochs, learning_rate, inputs, targets);
  

    predict_save(net);

    net.save("../models/model.txt");
    
    
    return 0;
}