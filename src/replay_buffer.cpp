#include "../include/replay_buffer.hpp"
#include <algorithm>
#include <random>

Replay_buffer::Replay_buffer(size_t cap, unsigned seed)
    : capacity(cap), rng(seed) {}

void Replay_buffer::push(const Transition& t) {
    if (buffer.size() < capacity) {
        buffer.push_back(t);
    } else {        
        buffer.pop_front();
        buffer.push_back(t);
    }
}

std::vector<Transition> Replay_buffer::sample(size_t batch_size) {
    std::vector<Transition> batch;
    batch.reserve(batch_size);
    
    if (buffer.size() < batch_size) return batch;
        
    std::vector<size_t> indices(buffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (size_t i = 0; i < batch_size; ++i) {
        batch.push_back(buffer[indices[i]]);
    }
    return batch;
}