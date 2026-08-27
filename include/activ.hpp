#pragma once

#include <cmath>
#include <string>

class Activation {
    public:
        virtual double forward(double x) = 0;
        virtual double backward(double x) = 0;
        virtual ~Activation() = default;  
        virtual std::string get_name() const = 0;      
};


class Sigmoid : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;  
        std::string get_name() const override;       
};

class ReLU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class LeakyReLU : public Activation {
    private:
        double alpha;
    public:
        LeakyReLU(double a = 0.01);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class Linear : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class TanH : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class SELU : public Activation {
    private:
        double lambda;
        double alpha;
    public:
        SELU(double l = 1.0507009873554804934193349852946, double a = 1.6732632423543772848170429916717);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class Swish : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class Mish : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class LSLU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class GELU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class ELU : public Activation {
    private:
        double alpha;
    public:
        ELU(double a = 1.0);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class Softplus : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

class APLU : public Activation {
    private:
        double alpha;  
        double lambda; 
        double beta;   
        double gamma;  
    public:
        APLU(double a = 0.05, double l = 1.0, double b = 0.2, double g = 1.0);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override; 
};

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
        GBA(double eps = 0.05, double b = 1.5, double amp = 0.25, double c = 2.0);
        double forward(double x) override;
        double backward(double x) override;
        std::string get_name() const override;
                
        double get_epsilon() const { return epsilon; }
        double get_beta() const { return beta; }
        double get_amplitude() const { return amplitude; }
        double get_center() const { return center; }
};
