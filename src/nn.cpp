#include "nn.hpp"
#include <cmath>
#include <random>

namespace nn {

namespace functional {

// ReLU Activation Function (max(0, x))
Tensor relu(const Tensor& A) {
    Tensor out(A.shape(), A.requires_grad_);
    size_t size = out.storage()->size();

    // Forward pass: apply max(0, val)
    for (size_t i = 0; i < size; i++) {
        float val = A.storage()->data()[i];
        out.storage()->data()[i] = (val > 0.0f) ? val : 0.0f;
    }

    // Backward pass definition
    if (out.requires_grad_) {
        out.parents_ = {A}; // Link to parent for autograd graph
        out.backward_op_ = [A, out, size]() mutable {
            if (A.requires_grad_ && A.grad_) {
                // Derivative of ReLU is 1 if x > 0, else 0
                for (size_t i = 0; i < size; i++) {
                    if (A.storage()->data()[i] > 0.0f) A.grad_->data()[i] += out.grad_->data()[i];
                }
            }
        };
    }

    return out;
}

// Mean Squared Error Loss
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    Tensor out({1}, pred.requires_grad_);
    size_t size = pred.storage()->size();

    // Forward pass: Compute the mean of squared differences
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = pred.storage()->data()[i] - target.storage()->data()[i];
        sum += diff * diff;
    }
    out.storage()->data()[0] = sum / size;

    // Backward pass definition
    if (out.requires_grad_) {
        out.parents_ = {pred};
        out.backward_op_ = [pred, target, out, size]() mutable {
            if (pred.requires_grad_ && pred.grad_) {
                float grad_out = out.grad_->data()[0];
                float float_size = static_cast<float>(size);
                // Derivative of MSE: 2 * (pred - target) / N
                for (size_t i = 0; i < size; i++) {
                    float diff = pred.storage()->data()[i] - target.storage()->data()[i];
                    pred.grad_->data()[i] += grad_out * (2.0f * diff / float_size);
                }
            }
        };
    }

    return out;
}

// Sigmoid Activation Function (1 / (1 + e^-x))
Tensor sigmoid(const Tensor& A) {
    Tensor out(A.shape(), A.requires_grad_);
    size_t size = out.storage()->size();
    
    // Forward pass
    for (size_t i = 0; i < size; i++) {
        float val = A.storage()->data()[i];
        out.storage()->data()[i] = 1.0f / (1.0f + std::exp(-val));
    }

    // Backward pass definition
    if (out.requires_grad_) {
        out.parents_ = {A};
        out.backward_op_ = [A, out, size]() mutable {
            // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            for (size_t i = 0; i < size; i++) {
                float s = out.storage()->data()[i];
                A.grad_->data()[i] += out.grad_->data()[i] * (s * (1.0f - s));
            }
        };
    }

    return out;
}

} // namespace functional

// Linear Layer Initialization
Linear::Linear(size_t in_features, size_t out_features)
    : weight(Tensor({in_features, out_features}, true)), bias(Tensor({1, out_features}, true)) {
    // Initialize weights using He initialization-like normal distribution
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / in_features));

    for (size_t i = 0; i < weight.storage()->size(); i++) weight.storage()->data()[i] = d(gen);
    for (size_t i = 0; i < bias.storage()->size(); i++) bias.storage()->data()[i] = 0.0f;
}

// Linear Layer Forward Pass: X * W + b
Tensor Linear::forward(const Tensor& B) {
    // Uses the optimized tiled SIMD matrix multiplication
    Tensor out = B.matmul_tiled_simd(weight);
    return out.add(bias);
}

// Return trainable parameters
std::vector<Tensor> Linear::parameters() const { return {weight, bias}; }

} // namespace nn
