#include "include/network.hpp"
#include "include/activ.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib> 
#include <ctime> 

void net_train(Network& net, int epochs, float l_r, const af::array& inputs, const af::array& targets) {
    for (int epoch = 0; epoch <= epochs; ++epoch) {

        if (epoch > 0 && epoch % 10000 == 0) {
            std::cout << "old lr: " << l_r;
            l_r /= 4.0f;                 
            std::cout << "; new lr: " << l_r << std::endl;       
        }
                
        float loss = net.train(inputs, targets, l_r);
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", loss: " << loss << std::endl;
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
        
    af::array x_af = af::array(af::seq(-100.0f, 100.0f, 0.5f)); 
    x_af = af::transpose(x_af);
    
    af::array input_batch = x_af / 100.0f;
    af::array output_batch = net.predict(input_batch); 
        
    af::array final_outputs = output_batch * 10000.0f;
    
    int num_elements = x_af.elements();
    std::vector<float> host_x(num_elements);
    std::vector<float> host_out(num_elements);
        
    x_af.host(host_x.data());
    final_outputs.host(host_out.data());
    
    for (int i = 0; i < num_elements; ++i) {
        file << "Input: " << host_x[i] << " -> " << host_out[i] << std::endl;
    }   
    
    file.close();
}



int main() {        
    try {                
        af::info(); 
    } catch (const af::exception& e) {
        std::cerr << "ArrayFire error " << e.what() << std::endl;        
        return 1;
    }

    std::vector<float> host_inputs, host_targets;
    for (float x = -30.0f; x <= 30.0f; x += 0.5f) {
        host_inputs.push_back(x);
        host_targets.push_back(x * x);
    } 
        
    int num_samples = host_inputs.size();
    af::array inputs_af(1, num_samples, host_inputs.data());
    af::array targets_af(1, num_samples, host_targets.data());
    
    af::array batch_inputs = inputs_af / 100.0f;
    af::array batch_targets = targets_af / 10000.0f;
    
    Network net;        
    net_init(net); 
        
    float learning_rate = 0.1f;  
    int epochs = 2000;         
            
    net_train(net, epochs, learning_rate, batch_inputs, batch_targets);
    
    predict_save(net);
    
    return 0;
}
