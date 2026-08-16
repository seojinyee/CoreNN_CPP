#pragma once
#include "storage.hpp"
#include <vector>
#include <memory>
#include <functional>

// Core class representing multi-dimensional mathematical arrays
class Tensor {
private:
    std::shared_ptr<Storage> storage_; // Pointer to underlying data
    std::vector<size_t> shape_;        // Dimensions of the array (e.g., 2x3 matrix)
    std::vector<size_t> strides_;      // Step size needed to traverse different dimensions
    size_t offset_;                    // Starting index in memory for tensor views

    void compute_strides();            // Utility to calculate strides from shapes

public:
    // Autograd properties
    bool requires_grad_;                // Flag checking if we should track this tensor
    std::shared_ptr<Storage> grad_;     // Holds gradient data
    std::vector<Tensor> parents_;       // Upstream dependencies (computation graph nodes)
    std::function<void()> backward_op_; // Closure that executes derivative math for this step

    // Standard constructor
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    
    // View constructor (shares memory, used heavily for Transpose)
    Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, bool requires_grad = false);

    // Metadata Getters
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t offset() const { return offset_; }
    const std::shared_ptr<Storage>& storage() const { return storage_; }

    // Memory layout utilities
    size_t compute_flat_index(const std::vector<size_t>& indices) const;
    bool is_contiguous() const;
    Tensor contiguous() const;

    // Fast data accessors
    float& at(const std::vector<size_t>& indices) { return storage_->data()[compute_flat_index(indices)]; }
    const float& at(const std::vector<size_t>& indices) const { return storage_->data()[compute_flat_index(indices)]; }

    // Math operations
    Tensor transpose(size_t dim0, size_t dim1) const;
    Tensor add(const Tensor& B) const;

    // Different Matmul optimization implementations
    Tensor matmul_naive(const Tensor& B) const;
    Tensor matmul_ikj(const Tensor& B) const;
    Tensor matmul_tiled(const Tensor& B, size_t tile_size = 64) const;
    Tensor matmul_simd(const Tensor& B) const;
    Tensor matmul_tiled_simd(const Tensor& B, size_t tile_size = 64) const;

    // Autograd trigger
    void backward();
};
