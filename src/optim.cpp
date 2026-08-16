#include "optim.hpp"

// SGD Optimizer constructor
SGD::SGD(std::vector<Tensor*> params, float lr) : params_(params), lr_(lr) {}

// Apply a single optimization step (Weight Update)
void SGD::step() {
    for (Tensor* p : params_) {
        if (p->grad_) {
            size_t size = p->storage()->size();
            // Update rule: w = w - learning_rate * gradient
            for (size_t i = 0; i < size; i++) p->storage()->data()[i] -= lr_ * p->grad_->data()[i];
        }
    }
}

// Reset gradients to zero before the next backward pass
void SGD::zero_grad(){
    for (Tensor* p : params_) {
        if (p->grad_) {
            size_t size = p->grad_->size();
            for (size_t i = 0; i < size; i++) p->grad_->data()[i] = 0.0f;
        }
    }
}
