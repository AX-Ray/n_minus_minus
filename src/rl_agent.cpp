#include "../include/rl_agent.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

DQNAgent::DQNAgent(int state_dim, int action_size, double lr, double gamma, double eps, double eps_min, double eps_decay)
    : gamma(gamma), epsilon(eps), epsilon_min(eps_min),
      epsilon_decay(eps_decay), learning_rate(lr),
      action_size(action_size), state_dim(state_dim),
      memory(10000)  
{    
    q_network.add_layer(std::make_unique<Layer>(state_dim, 24, std::make_unique<ReLU>()));
    q_network.add_layer(std::make_unique<Layer>(24, 24, std::make_unique<ReLU>()));
    q_network.add_layer(std::make_unique<Layer>(24, action_size, std::make_unique<Linear>()));
    
    target_network.add_layer(std::make_unique<Layer>(state_dim, 24, std::make_unique<ReLU>()));
    target_network.add_layer(std::make_unique<Layer>(24, 24, std::make_unique<ReLU>()));
    target_network.add_layer(std::make_unique<Layer>(24, action_size, std::make_unique<Linear>()));
    update_target_network(); 
}

int DQNAgent::act(const Matrix& state) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    if (dis(gen) < epsilon) {        
        static std::uniform_int_distribution<> action_dis(0, action_size - 1);
        return action_dis(gen);
    } else {
        Matrix q_values = q_network.forward(state);        
        int best = 0;
        for (int i = 1; i < q_values.rows; ++i) {
            if (q_values(i, 0) > q_values(best, 0)) {
                best = i;
            }
        }
        return best;
    }
}

void DQNAgent::remember(const Matrix& state, int action, double reward,const Matrix& next_state, bool done) {
    memory.push({state, action, reward, next_state, done});
}

void DQNAgent::replay(int batch_size) {
    if (memory.size() < batch_size) return;

    auto batch = memory.sample(batch_size);
    for (const auto& transition : batch) {        
        Matrix q_current = q_network.forward(transition.state);
        double target;
        if (transition.done) {
            target = transition.reward;
        } else {            
            Matrix q_next = target_network.forward(transition.next_state);            
            double max_next = q_next(0, 0);
            for (int i = 1; i < q_next.rows; ++i) {
                if (q_next(i, 0) > max_next) max_next = q_next(i, 0);
            }
            target = transition.reward + gamma * max_next;
        }
        
        Matrix target_q = q_current; 
        target_q(transition.action, 0) = static_cast<float>(target);
        
        q_network.train(transition.state, target_q, learning_rate);
    }
    
    decay_epsilon();
}

void DQNAgent::update_target_network() {
    target_network.copy_weights_from(q_network);
}

void DQNAgent::decay_epsilon() {
    if (epsilon > epsilon_min) {
        epsilon *= epsilon_decay;
        if (epsilon < epsilon_min) epsilon = epsilon_min;
    }
}

int DQNAgent::best_action(const Matrix& state) {
    Matrix q_values = q_network.forward(state);
    int best = 0;
    for (int i = 1; i < q_values.rows; ++i) {
        if (q_values(i, 0) > q_values(best, 0)) best = i;
    }
    return best;
}   

void DQNAgent::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    file << "DQNAgent" << std::endl;
    file << state_dim << " " << action_size << std::endl;
    file << learning_rate << " " << gamma << " " << epsilon << " ";
    file << epsilon_min << " " << epsilon_decay << std::endl;
    
    q_network.save(file);
    file << std::endl;
    target_network.save(file);

    file.close();
    std::cout << "DQNAgent saved to " << filename << std::endl;
}

void DQNAgent::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    
    std::string header;
    file >> header;
    if (header != "DQNAgent") {
        std::cerr << "Invalid file format: expected 'DQNAgent', got '" << header << "'" << std::endl;
        file.close();
        return;
    }
    
    int saved_state_dim, saved_action_size;
    file >> saved_state_dim >> saved_action_size;
    if (!file) {
        std::cerr << "Error reading state/action dimensions" << std::endl;
        file.close();
        return;
    }
    
    if (saved_state_dim != state_dim || saved_action_size != action_size) {
        std::cerr << "State/action dimension mismatch: expected ("
                  << state_dim << "," << action_size << ") but got ("
                  << saved_state_dim << "," << saved_action_size << ")" << std::endl;
        file.close();
        return;
    }
    
    double lr, gamma, eps, eps_min, eps_decay;
    file >> lr >> gamma >> eps >> eps_min >> eps_decay;
    if (!file) {
        std::cerr << "Error reading hyperparameters" << std::endl;
        file.close();
        return;
    }
    
    learning_rate = lr;
    this->gamma = gamma;
    epsilon = eps;
    epsilon_min = eps_min;
    epsilon_decay = eps_decay;
    
    try {
        q_network.load(file);
        target_network.load(file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading networks: " << e.what() << std::endl;
        file.close();
        return;
    }

    file.close();
    std::cout << "DQNAgent loaded from " << filename << std::endl;
}