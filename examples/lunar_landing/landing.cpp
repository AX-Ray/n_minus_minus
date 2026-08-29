#include "../../include/matrix.hpp"
#include "../../include/rl_agent.hpp"
#include <cstdlib>
#include <tuple>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <algorithm>

const float DT = 0.01;
const float GRAVITY = -1.0f;
const float MAIN_THRUST = 2.0f;
const float SIDE_THRUST = 0.5f;
const float MAX_ANGULAR = 4.0f;
const float LANDING_SPEED = 0.5f;
const float LANDING_ANGLE = 0.1f;

struct Lander {
    float x, y;
    float vx, vy;
    float theta, omega;    
};

Lander reset() {
    Lander lander =  { (rand()%200-100)/100.0f, 1.0f + (rand()%50)/100.0f, 0.0f, 0.0f, (rand()%60-30)/100.0f, 0.0f };
    return lander;
}

std::tuple<Lander, float, bool> step(int action, Lander& lander) {
    float reward = 0.0f;
    bool done = false;

    reward = -0.001f; 

    if (action == 1) {
        lander.vy += MAIN_THRUST * DT;
        reward -= 0.01f; 
    } else if (action == 2) {
        lander.omega -= SIDE_THRUST * DT;
        reward -= 0.005f;
    } else if (action == 3) {
        lander.omega += SIDE_THRUST * DT;
        reward -= 0.005f;
    }

    lander.vy += GRAVITY * DT; //GRAVITY

    lander.omega = std::max(-MAX_ANGULAR, std::min(MAX_ANGULAR, lander.omega));

    lander.x += lander.vx * DT;
    lander.y += lander.vy * DT;
    lander.theta += lander.omega * DT;    

    if (lander.y > 5.0f) {
        reward -= 100.0f;
        done = true;
    }


    if (lander.y <= 0) {
        lander.y = 0;
        if (std::abs(lander.vy) > LANDING_SPEED) {
            reward -= 100.0f;
            done = true;
        } 
        else if (std::abs(lander.vy) <= LANDING_SPEED && std::abs(lander.theta) <= LANDING_ANGLE) {
            reward += 100.0f;
            done = true;
        }
        else if (std::abs(lander.theta) > LANDING_ANGLE) {
            reward -= 100.0f;
            done = true;
        }
    }

    if (lander.y > 0) {
        float alignment = 1.0f - std::abs(lander.theta) / M_PI;
        reward += 0.01f * alignment;
    }

    if (std::abs(lander.x) > 2.0f) {
        reward -= 100.0f;
        done = true;
    }

    return {lander, reward, done};
}

Matrix get_lander_matrix(Lander& lander) {        
    Matrix m(6,1);
    m(0,0) = lander.x / 2.0f;  
    m(1,0) = lander.y / 2.0f;
    m(2,0) = lander.vx / 5.0f;
    m(3,0) = lander.vy / 5.0f;
    m(4,0) = lander.theta / M_PI;
    m(5,0) = lander.omega / 5.0f;    
    return m;
}



void train_dqn(DQNAgent& agent, int episodes, double initial_lr, double gamma, 
               double eps_start, double eps_min, double eps_decay) {
    
    double lr = initial_lr;
    double prev_avg_reward = -1000.0;
    const int BATCH_SIZE = 64;
    const int TARGET_UPDATE = 10;
    const int LOG_INTERVAL = 1;
    const int MAX_STEPS = 2000;
    
    std::ofstream log_file("../../logs/lunar_training.csv");
    if (!log_file.is_open()) {
        std::cerr << "Cannot open logs/lunar_training.csv" << std::endl;
        return;
    }
    log_file << "Episode,AvgReward,Epsilon,Steps,LR\n";

    std::vector<double> rewards_history;

    std::cout << "Start training" << std::endl << std::endl;

    for (int episode = 0; episode < episodes; ++episode) {
        Lander lander = reset();
        Matrix state = get_lander_matrix(lander);
        
        double episode_reward = 0.0;
        int steps = 0;
        bool done = false;

        while (!done) {
            int action = agent.act(state);
            auto [next_lander, reward, step_done] = step(action, lander);
            done = step_done;
            lander = next_lander;
            Matrix next_state = get_lander_matrix(next_lander);

            agent.remember(state, action, reward, next_state, done);
            
            if (agent.get_memory_size() > BATCH_SIZE) {
                agent.replay(BATCH_SIZE);
            }

            state = next_state;
            episode_reward += reward;
            steps++;            

            if (steps > MAX_STEPS) {
                done = true;
                episode_reward -= 100.0f;
                break;
            }
        }

        rewards_history.push_back(episode_reward);

        if ((episode + 1) % TARGET_UPDATE == 0) {
            agent.update_target_network();
        }

        if ((episode + 1) % LOG_INTERVAL == 0) {
            double avg_reward = 0.0;
            int start = std::max(0, (int)rewards_history.size() - LOG_INTERVAL);
            for (int i = start; i < (int)rewards_history.size(); ++i) {
                avg_reward += rewards_history[i];
            }
            avg_reward /= (rewards_history.size() - start);

            if (episode > 0 && episode % 3 == 0) {
                double adapt = prev_avg_reward / avg_reward;
                if (avg_reward <= prev_avg_reward) {
                    lr *= std::min(adapt, 0.85);
                } else {
                    lr *= std::clamp(adapt, 1.1, 1.35);
                }
                agent.set_learning_rate(lr);
                prev_avg_reward = avg_reward;
            }

            std::cout << "Episode " << std::setw(4) << (episode + 1) 
                      << "/" << episodes
                      << " | AvgReward: " << std::fixed << std::setprecision(2) << avg_reward
                      << " | Epsilon: " << std::setprecision(3) << agent.get_epsilon()
                      << " | Steps: " << steps
                      << " | LR: " << std::scientific << std::setprecision(4) << lr << std::endl;

            log_file << episode + 1 << ","
                     << avg_reward << ","
                     << agent.get_epsilon() << ","
                     << steps << ","
                     << lr << "\n";
            log_file.flush();
        } 
        if ((episode + 1) % LOG_INTERVAL == 75) {
            agent.decay_epsilon();
        }       
    }

    

    log_file.close();
    std::cout << std::endl << "Training complete" << std::endl;
}

int main() {
    std::cout << "Started" << std::endl;
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::cout << "Creating agent..." << std::endl;

    const int STATE_DIM = 6;
    const int ACTION_SIZE = 4;
    const double LEARNING_RATE = 0.001;
    const double GAMMA = 0.99;
    const double EPSILON_START = 1.0;
    const double EPSILON_MIN = 0.01;
    const double EPSILON_DECAY = 0.99999;
    const int EPISODES = 200;

    DQNAgent agent(STATE_DIM, ACTION_SIZE, LEARNING_RATE, GAMMA, 
                   EPSILON_START, EPSILON_MIN, EPSILON_DECAY);
    
    agent.build_architecture({        
        {128, "ReLU"},
        {128, "ReLU"},
        {4, "Linear"}
    });

    std::cout << "Agent created." << std::endl;
    std::cout << "Training DQN agent on Lunar Lander..." << std::endl;

    train_dqn(agent, EPISODES, LEARNING_RATE, GAMMA, 
              EPSILON_START, EPSILON_MIN, EPSILON_DECAY);

    agent.save("../../models/lunar_model.txt");    

    std::cout << std::endl << " Testing" << std::endl;
    Lander test_lander = reset();
    Matrix test_state = get_lander_matrix(test_lander);
    bool test_done = false;
    int test_steps = 0;
    double test_reward = 0.0;

    while (!test_done && test_steps < 1000) {
        int action = agent.best_action(test_state);
        auto [next_lander, reward, step_done] = step(action, test_lander);
        test_done = step_done;
        test_lander = next_lander;
        Matrix next_state = get_lander_matrix(next_lander);

        test_state = next_state;
        test_reward += reward;
        test_steps++;

        std::cout << "Step " << std::setw(3) << test_steps 
                  << " | x: " << std::fixed << std::setprecision(3) << test_lander.x 
                  << " y: " << test_lander.y
                  << " vy: " << test_lander.vy
                  << " theta: " << test_lander.theta
                  << " reward: " << reward << std::endl;
    }

    std::cout << std::endl << "Test finished. Total reward: " << test_reward 
              << ", steps: " << test_steps 
              << ", success: " << (test_done && test_lander.y == 0 && std::abs(test_lander.vy) <= LANDING_SPEED ? "YES" : "NO") 
              << std::endl;

    return 0;
}