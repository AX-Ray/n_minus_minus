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


double LSLU::forward(double x) {    
    double swish = x / (1.0 + std::exp(-x));
    double softplus = (x > 20.0) ? x : std::log1p(std::exp(x)); 
    return swish + softplus - std::log(2.0);
}

double LSLU::backward(double x) {    
    double sig = 1.0 / (1.0 + std::exp(-x));
    double d_swish = x * sig + sig * (1.0 - x * sig);
    double d_softplus = sig; 
    
    return d_swish + d_softplus;
}


double GELU::forward(double x) {
    constexpr double kSqrt2OverPi = 0.7978845608028654; 
    double x_cubed = x * x * x;
    double inner = kSqrt2OverPi * (x + 0.044715 * x_cubed);
    return 0.5 * x * (1.0 + std::tanh(inner));
}

double GELU::backward(double x) {
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    double x_sq = x * x;
    double x_cubed = x_sq * x;
    double inner = kSqrt2OverPi * (x + 0.044715 * x_cubed);
    double tanh_inner = std::tanh(inner);
        
    double sech_sq = 1.0 - tanh_inner * tanh_inner;
    double d_inner = kSqrt2OverPi * (1.0 + 3.0 * 0.044715 * x_sq);
    
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_sq * d_inner;
}


ELU::ELU(double a) : alpha(a) {}

double ELU::forward(double x) {
    return x >= 0.0 ? x : alpha * (std::exp(x) - 1.0);
}

double ELU::backward(double x) {    
    return x >= 0.0 ? 1.0 : alpha * std::exp(x);
}


double Softplus::forward(double x) {    
    return x > 20.0 ? x : std::log1p(std::exp(x));
}

double Softplus::backward(double x) {    
    return 1.0 / (1.0 + std::exp(-x));
}


APLU::APLU(double a, double l, double b, double g) : alpha(a), lambda(l), beta(b), gamma(g) {}

double APLU::forward(double x) {    
    double base = (x >= 0.0) ? (lambda * x) : (alpha * x);
    return base + beta * std::tanh(gamma * x);
}

double APLU::backward(double x) {    
    double d_base = (x >= 0.0) ? lambda : alpha;
        
    double t = std::tanh(gamma * x);
    double d_tanh = beta * gamma * (1.0 - t * t);
    
    return d_base + d_tanh;
}

