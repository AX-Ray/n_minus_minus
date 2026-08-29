#pragma once

#include <cmath>
#include <string>

/**
 * @brief Base abstract class for all activation functions.
 * 
 */
class Activation {
    public:
        /**
         * @brief Forward pass of the activation function.
         * @param x Input value.
         * @return Activated output.
         */
        virtual double forward(double x) = 0;
        
        /**
         * @brief Backward pass (derivative) of the activation function.
         * @param x Input value (usually the pre-activated value z).
         * @return Derivative of the activation function at x.
         */
        virtual double backward(double x) = 0;
        
        virtual ~Activation() = default;
        
        /**
         * @brief Get the name of the activation function for serialization.
         * @return String identifier of the activation.
         */
        virtual std::string get_name() const = 0;
};


// STANDARD ACTIVATIONS

/**
 * @brief Sigmoid activation function.
 * 
 *  Can suffer from vanishing gradients. Commonly used in binary classification output layers.
 */
class Sigmoid : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Rectified Linear Unit (ReLU) activation.
 * 
 * Non-linear, sparse activation, computationally cheap.
 * Can suffer from dead neurons (negative values become zero).
 */
class ReLU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Leaky ReLU activation.   
 * 
 *  Allows small gradient for negative values, preventing dead neurons.
 * Default alpha = 0.01.
 */
class LeakyReLU : public Activation {
    private:
        double alpha;
    public:
        /**
         * @param a Slope for negative values (default 0.01).
         */
        LeakyReLU(double a = 0.01);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Linear (identity) activation.
 * 
 * No non-linearity. Used in output layers for regression tasks.
 */
class Linear : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Hyperbolic tangent (TanH) activation.
 * 
 * Output in (-1,1), zero-centered, smooth. Often used in
 * hidden layers of RNNs and LSTMs.
 */
class TanH : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Scaled Exponential Linear Unit (SELU) activation.
 *
 * Self-normalizing, maintains mean zero and variance one.
 * Used with LeCun initialization. Default lambda=1.0507, alpha=1.6733.
 */
class SELU : public Activation {
    private:
        double lambda;
        double alpha;
    public:
        /**
         * @param l Lambda parameter (default 1.0507).
         * @param a Alpha parameter (default 1.6733).
         */
        SELU(double l = 1.0507009873554804934193349852946, double a = 1.6732632423543772848170429916717);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

// MODERN ACTIVATIONS 

/**
 * @brief Swish activation function (also known as SiLU).
 * 
 * Smooth, non-monotonic, unbounded above, bounded below.
 * Discovered by Google, often outperforms ReLU.
 */
class Swish : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Mish activation function.
 * 
 * Smooth, non-monotonic, unbounded above, bounded below.
 */
class Mish : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Gaussian Error Linear Unit (GELU) activation.
 * 
 * Smooth, non-monotonic, stochastic regularization.
 * Used in almost all modern Transformer architectures.
 */
class GELU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Exponential Linear Unit (ELU) activation.
 * 
 * Smooth for negative values, avoids dead neurons.
 * Default alpha = 1.0.
 */
class ELU : public Activation {
    private:
        double alpha;
    public:
        /**
         * @param a Alpha parameter for negative saturation (default 1.0).
         */
        ELU(double a = 1.0);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief Softplus activation function.
 * 
 * Smooth approximation of ReLU. Computationally more expensive.
 * Often used in probability models and variational autoencoders.
 */
class Softplus : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

// EXPERIMENTAL ACTIVATIONS 

/**
 * @brief LSLU 
 *  
 * 
 * Smooth, unbounded above, saturates to -ln(2) for negative x.
 * This is an experimental function created by combining Swish and Softplus.
 * It has not been studied in peer-reviewed literature.
 * 
 * @warning Not benchmarked. No theoretical guarantees.
 */
class LSLU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief APLU  
 * 
 * Four adaptive parameters (alpha, lambda, beta, gamma).
 * In this implementation, parameters are fixed at construction. 
 * 
 * @warning Not benchmarked. No theoretical guarantees.
 */
class APLU : public Activation {
    private:
        double alpha;  
        double lambda; 
        double beta;   
        double gamma;  
    public:
        /**
         * @param a Alpha (negative slope).
         * @param l Lambda (positive slope).
         * @param b Beta (amplitude of tanh component).
         * @param g Gamma (sharpness of tanh component).
         */
        APLU(double a = 0.05, double l = 1.0, double b = 0.2, double g = 1.0);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
};

/**
 * @brief GBA 
 * 
 * Combines linear, softplus, and tanh components.
 * 
 * Smooth, parameterized with 4 fixed parameters.
 * Uses centered softplus for improved gradient flow.
 * 
 * @warning Not benchmarked. No theoretical guarantees.
 */
class GBA : public Activation {
    private:
        double epsilon;
        double beta;
        double amplitude;
        double center;
        double tanh_center;
    
        double sigmoid(double x) const;
        double softplus_centered(double x) const;
        
    public:
        /**
         * @param eps Epsilon (linear skip connection weight, 0<eps<1).
         * @param b Beta (scaling of softplus).
         * @param amp Amplitude of tanh bridge component.
         * @param c Center of tanh bridge.
         */
        GBA(double eps = 0.05, double b = 1.5, double amp = 0.25, double c = 2.0);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
        
        // Getters for hyperparameters
        double get_epsilon() const { return epsilon; }
        double get_beta() const { return beta; }
        double get_amplitude() const { return amplitude; }
        double get_center() const { return center; }
};