#include "../include/activ.hpp"

double Sigmoid::forward(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::backward(double x) {
    return x * (1.0 - x);
}

double ReLU::forward(double x) {
    return std::max(0.0, x);
}

double ReLU::backward(double x) {
    return x > 0 ? 1.0 : 0.0;
}

LeakyReLU::LeakyReLU(double a) : alpha(a) {}

double LeakyReLU::forward(double x) {
    return x > 0 ? x : alpha * x;
}

double LeakyReLU::backward(double x) {
    return x > 0 ? 1.0 : alpha;
}

double Linear::forward(double x) {
    return x;
}

double Linear::backward(double x) {
    return 1.0;
}