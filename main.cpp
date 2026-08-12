#include "include/network.hpp"
#include "include/activ.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib> 
#include <ctime> 
#include <algorithm>

void net_train(Network& net, int epochs, double l_r, const std::vector<double>& inputs, const std::vector<double>& targets) {
    double prev_total_loss = 999999.0; 

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
                
        for (int i = 0; i < inputs.size(); i++) {
            Matrix input(1, 1);
            input(0, 0) = inputs[i] / 100.0; 
            
            Matrix target(1, 1);
            target(0, 0) = targets[i] / 10000.0;  
            
            total_loss += net.train(input, target, l_r);
        }  
        
        double current_epoch_loss = total_loss / inputs.size();

        if (epoch > 0 && epoch % 100 == 0) {   
                        
            std::ofstream file("logs/log.txt", std::ios::app); 
            
            if (file.is_open()) {
                file << current_epoch_loss << " " << l_r <<  std::endl;
                file.close(); 
            } else {
                std::cerr << "Cannot open log.txt" << std::endl;
            }
        }
        
        if (epoch > 0 && epoch % 500 == 0) {            
            double adapt = prev_total_loss / current_epoch_loss;
            if (current_epoch_loss >= prev_total_loss) {                
                l_r *= std::min(adapt, 0.85);            
            } else {                      
                l_r *= std::clamp(adapt, 1.1, 1.35); 
            }
            prev_total_loss = current_epoch_loss; 
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", loss: " << current_epoch_loss << " (LR: " << l_r << ")" << std::endl;
        }
    }
}


void net_init(Network& net) { 
    net.add_layer(std::make_unique<Layer>(1, 20, std::make_unique<LSLU>()));       
    net.add_layer(std::make_unique<Layer>(20, 20, std::make_unique<LSLU>()));       
    net.add_layer(std::make_unique<Layer>(20, 1, std::make_unique<Linear>()));       
}

void predict_save(Network& net, const std::string& filename = "output/predict.txt") {
    std::ofstream file(filename);
        
    for (double x = -100.0; x <= 100.0; x += 0.5) {
        Matrix input(1, 1);
        input(0, 0) = x / 100.0;  
        
        Matrix output = net.predict(input);        

        file << "Input: " << x << " -> " << output(0, 0) * 10000.0 << std::endl;
    }   
    
    file.close();
}


int main() {    
    std::srand(time(nullptr)); 

    std::ofstream clear_file("logs/log.txt", std::ios::trunc);
    clear_file.close(); 

    Network net;        
    net_init(net);
        
    std::vector<double> inputs, targets;
    for (double x = -30.0; x <= 30.0; x += 0.5) {
        inputs.push_back(x);
        targets.push_back(x * x);
    } 
        
    double learning_rate = 0.2;  
    int epochs = 30000;         
        
    
    net_train(net, epochs, learning_rate, inputs, targets);
  

    predict_save(net);
    
    
    return 0;
}