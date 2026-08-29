#include "activ.hpp"
#include <vector>

std::unique_ptr<Activation> create_activation(const std::string& name) {
        if (name == "Sigmoid") return std::make_unique<Sigmoid>();
        if (name == "ReLU") return std::make_unique<ReLU>();
        if (name == "LeakyReLU") return std::make_unique<LeakyReLU>();
        if (name == "Linear") return std::make_unique<Linear>();
        if (name == "TanH") return std::make_unique<TanH>();
        if (name == "SELU") return std::make_unique<SELU>();
        if (name == "Swish") return std::make_unique<Swish>();
        if (name == "Mish") return std::make_unique<Mish>();
        if (name == "LSLU") return std::make_unique<LSLU>();
        if (name == "GELU") return std::make_unique<GELU>();
        if (name == "ELU") return std::make_unique<ELU>();
        if (name == "Softplus") return std::make_unique<Softplus>();
        if (name == "APLU") return std::make_unique<APLU>();
        throw std::runtime_error("Unknown activation name: " + name);
}


inline std::vector<std::string> get_all_activation_names() {
    return {
        "Sigmoid", "ReLU", "LeakyReLU", "Linear", "TanH",
        "SELU", "Swish", "Mish", "LSLU", "GELU",
        "ELU", "Softplus", "APLU", "GBA"
    };
}