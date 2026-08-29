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

    double get_epsilon() const {return epsilon; }
    double get_gamma() const { return gamma; }
    double get_learning_rate() const { return learning_rate; }
    int get_state_dim() const { return state_dim; }
    int get_action_size() const { return action_size; }
    Network& get_q_network() { return q_network; }
    const Network& get_q_network() const { return q_network; }
    Network& get_target_network() { return target_network; }
    const Network& get_target_network() const { return target_network; }
    
    void set_epsilon(double eps) { epsilon = eps; }
    void set_gamma(double g) { gamma = g; }
    void set_learning_rate(double lr) { learning_rate = lr; }
    
    int best_action(const Matrix& state);  
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};