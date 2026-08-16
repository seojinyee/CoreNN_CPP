#pragma once
#include "tensor.hpp"
#include <vector>

namespace nn {

// Functional namespace containing stateless operations
namespace functional {
    Tensor relu(const Tensor& A);
    Tensor mse_loss(const Tensor& pred, const Tensor& target);
    Tensor sigmoid(const Tensor& input);
}

// Base class for neural network layers containing parameters
class Module {
public:
    virtual ~Module() = default;
    // Returns all weights/biases managed by this module
    virtual std::vector<Tensor> parameters() const = 0;
};

// Standard Dense/Fully-Connected Layer (Y = XW + b)
class Linear : public Module {
public:
    Tensor weight;
    Tensor bias;

    Linear(size_t in_features, size_t out_features);
    Tensor forward(const Tensor& A);

    // Operator overload to mimic function call syntax (e.g., output = layer(input))
    Tensor operator()(const Tensor& A) { return forward(A); }

    std::vector<Tensor> parameters() const override;
};

} // namespace nn
