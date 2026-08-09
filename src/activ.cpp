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


double TanH::forward(double x) {
    return std::tanh(x);
}

double TanH::backward(double x) {    
    return 1.0 - x * x;
}


SELU::SELU(double l, double a) : lambda(l), alpha(a) {}

double SELU::forward(double x) {
    return x >= 0 ? lambda * x : lambda * alpha * (std::exp(x) - 1.0);
}

double SELU::backward(double x) {    
    return x >= 0 ? lambda : x + lambda * alpha;
}


double Swish::forward(double x) {    
    return x / (1.0 + std::exp(-x));
}

double Swish::backward(double x) {    
    double sig = 1.0 / (1.0 + std::exp(-x));
    return x * sig + sig * (1.0 - x * sig);
}


double Mish::forward(double x) {    
    double sp = std::log(1.0 + std::exp(x));  
    return x * std::tanh(sp);
}

double Mish::backward(double x) {    
    double ex = std::exp(x);
    double ex2 = ex * ex;
    double ex3 = ex2 * ex;
    
    double omega = 4.0 * (x + 1.0) + 4.0 * ex2 + ex3 + ex * (4.0 * x + 6.0);
    double delta = 2.0 * ex + ex2 + 2.0;
    
    return (ex * omega) / (delta * delta);
}