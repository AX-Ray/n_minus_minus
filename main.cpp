#include "include/network.hpp"
#include "include/activ.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>

int main() {    
    std::srand(time(nullptr)); 


    Network net;    
    net.add_layer(std::make_unique<Layer>(1, 15, std::make_unique<LeakyReLU>()));  
    net.add_layer(std::make_unique<Layer>(15, 15, std::make_unique<LeakyReLU>()));
    net.add_layer(std::make_unique<Layer>(15, 1, std::make_unique<Linear>()));       
        
    std::vector<double> inputs;
    for (double x = -300.0; x <= 300.0; x += 0.5) {
        inputs.push_back(x);
    }
    std::vector<double> targets;
    for (double x : inputs) {
        targets.push_back(x * 2.0);  
    }    
        
    double learning_rate = 0.01;  
    int epochs = 30000; 
    
    net.load("model_mult2v4.txt");
        
    
    /*for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (int i = 0; i < inputs.size(); i++) {
            Matrix input(1, 1);
            input(0, 0) = inputs[i] / 100.0; 
            
            Matrix target(1, 1);
            target(0, 0) = targets[i] / 100.0;  
            
            total_loss += net.train(input, target, learning_rate);
        }  
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    std::string filename = "model_mult2v4.txt";
    net.save(filename);   */


    std::ofstream file("predict.txt");
        
    for (double x = -100000.0; x <= 100000.0; x += 0.5) {
        Matrix input(1, 1);
        input(0, 0) = x / 100.0;  
        
        Matrix output = net.predict(input);        

        file << "Input: " << x << " -> " << output(0, 0) * 100.0 << std::endl;
    }   
    
    file.close();
    
    return 0;
}