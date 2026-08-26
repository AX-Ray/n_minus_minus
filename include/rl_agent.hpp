#pragma once
#include "network.hpp"
#include "replay_buffer.hpp"

class DQNAgent {
    Network q_network;     
    Network target_network; 
    Replay_buffer memory;
    double gamma;           
    double epsilon;        
    double epsilon_min;
    double epsilon_decay;
    double learning_rate;
    int action_size;
    int state_dim;
public:
    DQNAgent(int state_dim, int action_size, double lr, double gamma, double eps = 1.0, double eps_min = 0.01, double eps_decay = 0.995);
    
    int act(const Matrix& state);  
    void remember(const Matrix& state, int action, double reward, const Matrix& next_state, bool done);
    void replay(int batch_size);    
    void update_target_network();  
    void decay_epsilon();

    double get_epsilon() const;

    int best_action(const Matrix& state);  
};