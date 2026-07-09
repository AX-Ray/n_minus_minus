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