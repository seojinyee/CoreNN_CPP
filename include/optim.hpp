#pragma once
#include "tensor.hpp"
#include <vector>

// Stochastic Gradient Descent Optimizer class
class SGD {
private:
    std::vector<Tensor*> params_; // Pointers to network parameters that need updating
    float lr_;                    // Learning Rate (step size for updates)

public:
    SGD(std::vector<Tensor*> params, float lr);

    // Applies weight updates using accumulated gradients
    void step();
    // Resets gradient arrays to zero
    void zero_grad();
};
