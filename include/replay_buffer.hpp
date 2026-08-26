#pragma once
#include <vector>
#include <deque>
#include <random>
#include "matrix.hpp"

struct Transition {
    Matrix state;
    int action;
    double reward;
    Matrix next_state;
    bool done;
};

class Replay_buffer {
    size_t capacity;
    std::deque<Transition> buffer;
    std::mt19937 rng;
public:
    Replay_buffer(size_t cap, unsigned seed = std::random_device{}());
    void push(const Transition& t);
    std::vector<Transition> sample(size_t batch_size);
    size_t size() const { return buffer.size(); }
};