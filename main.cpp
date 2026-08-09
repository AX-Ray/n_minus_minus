#include "include/network.hpp"
#include "include/activ.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib> 
#include <ctime> 

void net_train(Network& net, int epochs, double l_r, const std::vector<double>& inputs, const std::vector<double>& targets) {
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;

        if (epoch > 0 && epoch % 10000 == 0) { // Changing learning rate during trainig
            l_r /= 4.0;            
        }
        
        for (int i = 0; i < inputs.size(); i++) {
            Matrix input(1, 1);
            input(0, 0) = inputs[i] / 100.0; 
            
            Matrix target(1, 1);
            target(0, 0) = targets[i] / 10000.0;  
            
            total_loss += net.train(input, target, l_r);
        }  
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", loss: " << total_loss / inputs.size() << std::endl;
        }
    }
}

void net_init(Network& net) { 
    net.add_layer(std::make_unique<Layer>(1, 20, std::make_unique<ReLU>()));       
    net.add_layer(std::make_unique<Layer>(20, 20, std::make_unique<LeakyReLU>(0.01)));       
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

    Network net;        
    net_init(net);
        
    std::vector<double> inputs, targets;
    for (double x = -30.0; x <= 30.0; x += 0.5) {
        inputs.push_back(x);
        targets.push_back(x * x);
    } 
        
    double learning_rate = 0.1;  
    int epochs = 10000;         
        
    
    net_train(net, epochs, learning_rate, inputs, targets);
  

    predict_save(net);
    
    
    return 0;
}