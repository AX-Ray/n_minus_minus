#include "../include/activ.hpp"
#include <cmath>

static af::array af_sigmoid(const af::array& x) {
    return 1.0f / (1.0f + af::exp(-x));
}


af::array Sigmoid::forward(const af::array& x) {
    return af_sigmoid(x);
}

af::array Sigmoid::backward(const af::array& x) {    
    return x * (1.0f - x);
}


af::array ReLU::forward(const af::array& x) {    
    return af::max(x, 0.0f);
}

af::array ReLU::backward(const af::array& x) {    
    return (x > 0.0f).as(f32);
}


LeakyReLU::LeakyReLU(float a) : alpha(a) {}

af::array LeakyReLU::forward(const af::array& x) {    
    return af::select(x > 0.0f, x, alpha * x);
}

af::array LeakyReLU::backward(const af::array& x) {        
    af::array true_branch = af::constant(1.0f, x.dims(), f32);
    af::array false_branch = af::constant(alpha, x.dims(), f32);
        
    return af::select(x > 0.0f, true_branch, false_branch);
}



af::array Linear::forward(const af::array& x) {
    return x;
}

af::array Linear::backward(const af::array& x) {    
    return af::constant(1.0f, x.dims(), f32);
}


af::array TanH::forward(const af::array& x) {
    return af::tanh(x);
}

af::array TanH::backward(const af::array& x) {    
    return 1.0f - x * x;
}


SELU::SELU(float l, float a) : lambda(l), alpha(a) {}

af::array SELU::forward(const af::array& x) {
    return af::select(x >= 0.0f, lambda * x, lambda * alpha * (af::exp(x) - 1.0f));
}

af::array SELU::backward(const af::array& x) {        
    return af::select(x >= 0.0f, lambda, lambda * alpha * af::exp(x));
}


af::array Swish::forward(const af::array& x) {    
    return x * af_sigmoid(x);
}

af::array Swish::backward(const af::array& x) {    
    af::array sig = af_sigmoid(x);
    return x * sig + sig * (1.0f - x * sig);
}


af::array Mish::forward(const af::array& x) {        
    af::array sp = af::log1p(af::exp(x));  
    return x * af::tanh(sp);
}

af::array Mish::backward(const af::array& x) {    
    af::array ex = af::exp(x);
    af::array ex2 = ex * ex;
    af::array ex3 = ex2 * ex;
    
    af::array omega = 4.0f * (x + 1.0f) + 4.0f * ex2 + ex3 + ex * (4.0f * x + 6.0f);
    af::array delta = 2.0f * ex + ex2 + 2.0f;
    
    return (ex * omega) / (delta * delta);
}


af::array LSLU::forward(const af::array& x) {    
    af::array swish = x * af_sigmoid(x);    
    af::array softplus = af::select(x > 20.0f, x, af::log1p(af::exp(x))); 
    return swish + softplus - std::log(2.0f);
}

af::array LSLU::backward(const af::array& x) {    
    af::array sig = af_sigmoid(x);
    af::array d_swish = x * sig + sig * (1.0f - x * sig);
    af::array d_softplus = sig; 
    return d_swish + d_softplus;
}


af::array SLSLU::forward(const af::array& x) {    
    af::array swish = x * af_sigmoid(x);
    af::array softplus = af::select(x > 20.0f, x, af::log1p(af::exp(x))); 
    return 0.5f * (swish + softplus - std::log(2.0f));
}

af::array SLSLU::backward(const af::array& x) {    
    af::array sig = af_sigmoid(x);
    af::array d_swish = x * sig + sig * (1.0f - x * sig);
    af::array d_softplus = sig; 
    return 0.5f * (d_swish + d_softplus);
}


af::array GELU::forward(const af::array& x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f; 
    af::array x_cubed = x * x * x;
    af::array inner = kSqrt2OverPi * (x + 0.044715f * x_cubed);
    return 0.5f * x * (1.0f + af::tanh(inner));
}

af::array GELU::backward(const af::array& x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    af::array x_sq = x * x;
    af::array x_cubed = x_sq * x;
    af::array inner = kSqrt2OverPi * (x + 0.044715f * x_cubed);
    af::array tanh_inner = af::tanh(inner);
        
    af::array sech_sq = 1.0f - tanh_inner * tanh_inner;
    af::array d_inner = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x_sq);
    
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_sq * d_inner;
}


ELU::ELU(float a) : alpha(a) {}

af::array ELU::forward(const af::array& x) {
    return af::select(x >= 0.0f, x, alpha * (af::exp(x) - 1.0f));
}

af::array ELU::backward(const af::array& x) {    
    return af::select(x >= 0.0f, 1.0f, alpha * af::exp(x));
}


af::array Softplus::forward(const af::array& x) {    
    return af::select(x > 20.0f, x, af::log1p(af::exp(x)));
}

af::array Softplus::backward(const af::array& x) {    
    return af_sigmoid(x);
}


APLU::APLU(float a, float l, float b, float g) : alpha(a), lambda(l), beta(b), gamma(g) {}

af::array APLU::forward(const af::array& x) {    
    af::array base = af::select(x >= 0.0f, lambda * x, alpha * x);
    return base + beta * af::tanh(gamma * x);
}

af::array APLU::backward(const af::array& x) {        
    af::array true_branch = af::constant(lambda, x.dims(), f32);
    af::array false_branch = af::constant(alpha, x.dims(), f32);
        
    af::array d_base = af::select(x >= 0.0f, true_branch, false_branch);
        
    af::array t = af::tanh(gamma * x);
    af::array d_tanh = beta * gamma * (1.0f - t * t);
    
    return d_base + d_tanh;
}



ELUSwish::ELUSwish(float a, float b) : alpha(a), beta(b) {}

af::array ELUSwish::forward(const af::array& x) {
    af::array sig_pos = af_sigmoid(x);      
    af::array sig_neg = 1.0f / (1.0f + af::exp(beta * x)); 
    af::array neg_part = (af::exp(-beta * x) - 1.0f) * sig_neg;
    return x * sig_pos + alpha * neg_part;
}

af::array ELUSwish::backward(const af::array& x) {    
    af::array sig_pos = af_sigmoid(x);
    af::array d_swish = sig_pos + x * sig_pos * (1.0f - sig_pos);
    
    af::array sig_neg = 1.0f / (1.0f + af::exp(beta * x));
    af::array exp_neg = af::exp(-beta * x);
    af::array u = exp_neg - 1.0f;
    af::array d_u = -beta * exp_neg;
    af::array dsig_neg = -beta * sig_neg * (1.0f - sig_neg);
    af::array d_neg = d_u * sig_neg + u * dsig_neg;

    return d_swish + alpha * d_neg;
}
