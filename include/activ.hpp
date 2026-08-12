#pragma once

#include <cmath>

class Activation {
    public:
        virtual double forward(double x) = 0;
        virtual double backward(double x) = 0;
        virtual ~Activation() = default;        
};


class Sigmoid : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;        
};

class ReLU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class LeakyReLU : public Activation {
    private:
        double alpha;
    public:
        LeakyReLU(double a = 0.01);
        double forward(double x) override;
        double backward(double x) override;
};

class Linear : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class TanH : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class SELU : public Activation {
    private:
        double lambda;
        double alpha;
    public:
        SELU(double l = 1.0507009873554804934193349852946, double a = 1.6732632423543772848170429916717);
        double forward(double x) override;
        double backward(double x) override;
};

class Swish : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class Mish : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class LSLU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class GELU : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
};

class ELU : public Activation {
    private:
        double alpha;
    public:
        ELU(double a = 1.0);
        double forward(double x) override;
        double backward(double x) override;
};

class Softplus : public Activation {
    public:
        double forward(double x) override;
        double backward(double x) override;
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
};

