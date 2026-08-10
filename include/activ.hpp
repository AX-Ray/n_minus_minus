#pragma once

#include <arrayfire.h> 

class Activation {
    public:        
        virtual af::array forward(const af::array& x) = 0;
        virtual af::array backward(const af::array& x) = 0;
        virtual ~Activation() = default;
};


class Sigmoid : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class ReLU : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class LeakyReLU : public Activation {
    private:
        float alpha; 
    public:
        LeakyReLU(float a = 0.01f);
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class Linear : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class TanH : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class SELU : public Activation {
    private:
        float lambda;
        float alpha;
    public:        
        SELU(float l = 1.050701f, float a = 1.673263f);
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class Swish : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class Mish : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class LSLU : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class SLSLU : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class GELU : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class ELU : public Activation {
    private:
        float alpha;
    public:
        ELU(float a = 1.0f);
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class Softplus : public Activation {
    public:
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class APLU : public Activation {
    private:
        float alpha;  
        float lambda; 
        float beta;   
        float gamma;  
    public:
        APLU(float a = 0.05f, float l = 1.0f, float b = 0.2f, float g = 1.0f);
        af::array forward(const af::array& x) override;
        af::array backward(const af::array& x) override;
};

class ELUSwish : public Activation {
private:
    float alpha;
    float beta;
public:
    ELUSwish(float a = 1.0f, float b = 1.0f);
    af::array forward(const af::array& x) override;
    af::array backward(const af::array& x) override;
};
