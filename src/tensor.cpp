#include "tensor.hpp"
#include <numeric>
#include <immintrin.h>
#include <set>

// Calculates the strides required to traverse multi-dimensional data in a flat array
void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_t current_stride = 1;

    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        strides_[i] = current_stride;
        current_stride *= shape_[i];
    }
}

// Constructor for a new tensor allocating fresh memory
Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad) : shape_(shape), offset_(0), requires_grad_(requires_grad) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    storage_ = std::make_shared<Storage>(total_size);
    compute_strides();

    // Allocate memory for gradients if this tensor needs to be tracked
    if (requires_grad_) {
        grad_ = std::make_shared<Storage>(total_size);
        std::fill(grad_->data(), grad_->data() + total_size, 0.0f);
    }
}

// Constructor that shares existing memory (useful for operations like Transpose or Views)
Tensor::Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, bool requires_grad)
    : storage_(std::move(storage)), shape_(std::move(shape)), strides_(std::move(strides)), offset_(offset), requires_grad_(requires_grad) {
    if (requires_grad_) {
        size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
        grad_ = std::make_shared<Storage>(total_size);
        std::fill(grad_->data(), grad_->data() + total_size, 0.0f);
    }
}

// Converts multi-dimensional indices to a 1D flat array index using strides
size_t Tensor::compute_flat_index(const std::vector<size_t>& indices) const {
    size_t flat_idx = offset_;
    for (size_t i = 0; i < indices.size(); i++) flat_idx += indices[i] * strides_[i];
    return flat_idx;
}

// Checks if the tensor data is stored sequentially in memory
bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    size_t expected_stride = 1;

    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        if (strides_[i] != expected_stride) return false;
        expected_stride *= shape_[i];
    }

    return true;
}

// Forces a non-contiguous tensor into a new contiguous memory layout
Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    Tensor out(shape_, requires_grad_);
    size_t ndim = shape_.size();
    size_t total_elements = out.storage()->size();

    // Copy elements one by one based on physical layout
    for (size_t i = 0; i < total_elements; i++) {
        std::vector<size_t> coords(ndim);
        size_t temp = i;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            coords[d] = temp % shape_[d];
            temp /= shape_[d];
        }
        size_t orig_flat_idx = offset_;
        for (size_t d = 0; d < ndim; d++) orig_flat_idx += coords[d] * strides_[d];
        out.storage()->data()[i] = storage_->data()[orig_flat_idx];
    }

    // Pass gradients back if autograd is enabled
    if (requires_grad_) {
        Tensor self = *this;
        out.parents_ = {self};
        out.backward_op_ = [self, out, total_elements]() mutable {
            if (self.requires_grad_ && self.grad_) {
                for (size_t i = 0; i < total_elements; i++) self.grad_->data()[i] += out.grad_->data()[i];
            }
        };
    }

    return out;
}

// Zero-copy transpose: Just swap shapes and strides
Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    std::vector<size_t> new_shape = shape_;
    std::vector<size_t> new_strides = strides_;
    
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    
    Tensor out(storage_, new_shape, new_strides, offset_, requires_grad_);

    if (requires_grad_) {
        Tensor self = *this;
        out.parents_ = {self};

        // Handle backward for transpose by routing gradients to the correct original indices
        out.backward_op_ = [self, out, dim0, dim1]() mutable {
            if (self.requires_grad_ && self.grad_) {
                size_t ndim = out.shape_.size();
                size_t total_elements = out.grad_->size();

                std::vector<size_t> self_contig_strides(ndim);
                size_t current_stride = 1;
                for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                    self_contig_strides[d] = current_stride;
                    current_stride *= self.shape_[d];
                }

                for (size_t i = 0; i < total_elements; i++) {
                    float grad_val = out.grad_->data()[i];
                    
                    std::vector<size_t> coords(ndim);
                    size_t temp = i;
                    for (int d = static_cast<int>(ndim) - 1; d >= 0; d--) {
                        coords[d] = temp % out.shape_[d];
                        temp /= out.shape_[d];
                    }

                    std::swap(coords[dim0], coords[dim1]);

                    size_t self_flat_idx = 0;
                    for (size_t d = 0; d < ndim; d++) self_flat_idx += coords[d] * self_contig_strides[d];

                    self.grad_->data()[self_flat_idx] += grad_val;
                }
            }
        };
    }

    return out;
}

// Addition operation with simple broadcasting support
Tensor Tensor::add(const Tensor& B) const {
    // Operations assume contiguous memory for simplicity in this project
    if (!this->is_contiguous() || !B.is_contiguous()) return this->contiguous().add(B.contiguous());

    bool req_grad = requires_grad_ || B.requires_grad_;
    Tensor out(shape_, req_grad);

    size_t total_size = out.storage()->size();
    size_t b_size = B.storage()->size();
    bool broadcast = (b_size != total_size);

    for (size_t i = 0; i < total_size; i++) {
        size_t b_idx = broadcast ? (i % b_size) : i;
        out.storage()->data()[i] = storage_->data()[i] + B.storage()->data()[b_idx];
    }

    if (req_grad) {
        Tensor A = *this;
        out.parents_ = {A, B};

        // Gradients pass right through an addition operation
        out.backward_op_ = [A, B, out, broadcast, total_size, b_size]() mutable {
            for (size_t i = 0; i < total_size; i++) {
                if (A.requires_grad_ && A.grad_) A.grad_->data()[i] += out.grad_->data()[i];
                if (B.requires_grad_ && B.grad_) {
                    size_t b_idx = broadcast ? (i % b_size) : i;
                    B.grad_->data()[b_idx] += out.grad_->data()[i];
                }
            }
        };
    }
    
    return out;
}

// The slowest, standard nested loop matrix multiplication
Tensor Tensor::matmul_naive(const Tensor& B) const {
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = B.shape_[1];
    
    Tensor C({M, N});

    float* ptr_A = storage_->data() + offset_;
    float* ptr_B = B.storage_->data() + B.offset_;
    float* ptr_C = C.storage_->data() + C.offset_;

    size_t stride_A0 = strides_[0], stride_A1 = strides_[1];
    size_t stride_B0 = B.strides_[0], stride_B1 = B.strides_[1];
    size_t stride_C0 = C.strides_[0], stride_C1 = C.strides_[1];


    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) sum += ptr_A[i * stride_A0 + k * stride_A1] * ptr_B[k * stride_B0 + j * stride_B1];
            ptr_C[i * stride_C0 + j * stride_C1] = sum;
        }
    }

    return C;
}

// Improved Matmul: Reordered loops (i-k-j) for better cache access patterns
Tensor Tensor::matmul_ikj(const Tensor& B) const {
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = B.shape_[1];

    bool req_grad = requires_grad_ || B.requires_grad_;    
    Tensor C({M, N}, req_grad);

    float* ptr_A = storage_->data() + offset_;
    float* ptr_B = B.storage_->data() + B.offset_;
    float* ptr_C = C.storage_->data() + C.offset_;

    size_t stride_A0 = strides_[0], stride_A1 = strides_[1];
    size_t stride_B0 = B.strides_[0], stride_B1 = B.strides_[1];
    size_t stride_C0 = C.strides_[0], stride_C1 = C.strides_[1];

    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float a_ik = ptr_A[i * stride_A0 + k * stride_A1];
            for (size_t j = 0; j < N; j++) ptr_C[i * stride_C0 + j * stride_C1] += a_ik * ptr_B[k * stride_B0 + j * stride_B1];
        }
    }

    return C;
}

// Further optimization: Process small blocks (tiles) to keep data in CPU cache
Tensor Tensor::matmul_tiled(const Tensor& B, size_t tile_size) const {
    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = B.shape_[1];

    Tensor C({M, N});

    float* ptr_A = storage_->data() + offset_;
    float* ptr_B = B.storage_->data() + B.offset_;
    float* ptr_C = C.storage_->data() + C.offset_;

    size_t stride_A0 = strides_[0], stride_A1 = strides_[1];
    size_t stride_B0 = B.strides_[0], stride_B1 = B.strides_[1];
    size_t stride_C0 = C.strides_[0], stride_C1 = C.strides_[1];

    for (size_t i0 = 0; i0 < M; i0 += tile_size) {
        for (size_t k0 = 0; k0 < K; k0 += tile_size) {
            for (size_t j0 = 0; j0 < N; j0 += tile_size) {
                size_t i_max = std::min(i0 + tile_size, M);
                size_t k_max = std::min(k0 + tile_size, K);
                size_t j_max = std::min(j0 + tile_size, N);
            
                for (size_t i = i0; i < i_max; i++) {
                    for (size_t k = k0; k < k_max; k++) {
                        float a_ik = ptr_A[i * stride_A0 + k * stride_A1];
                        for(size_t j = j0; j < j_max; j++) ptr_C[i * stride_C0 + j * stride_C1] += a_ik * ptr_B[k * stride_B0 + j * stride_B1];
                    }    
                }
            }
        }
    }
    
    return C;    
}

// Hardware acceleration: Process 8 floats at a time using AVX2 registers
Tensor Tensor::matmul_simd(const Tensor& B) const {
    if (!this->is_contiguous() || !B.is_contiguous()) return this->contiguous().matmul_simd(B.contiguous());

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = B.shape_[1];

    Tensor C({M, N});

    float* ptr_A = storage_->data() + offset_;
    float* ptr_B = B.storage_->data() + B.offset_;
    float* ptr_C = C.storage_->data() + C.offset_;

    size_t stride_A0 = strides_[0], stride_A1 = strides_[1];
    size_t stride_B0 = B.strides_[0];
    size_t stride_C0 = C.strides_[0];

    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float a_ik = ptr_A[i * stride_A0 + k * stride_A1];
            // Load the single float value into a 256-bit register (duplicates it 8 times)
            __m256 a_vec = _mm256_set1_ps(a_ik);

            size_t j = 0;

            // Perform fused multiply-add (FMA) 8 elements at a time
            for (; j + 8 <= N; j+= 8) {
                __m256 c_vec = _mm256_loadu_ps(&ptr_C[i * stride_C0 + j]);
                __m256 b_vec = _mm256_loadu_ps(&ptr_B[k * stride_B0 + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec); // C = A * B + C
                _mm256_storeu_ps(&ptr_C[i * stride_C0 + j], c_vec);
            }

            // Handle remaining elements that don't fit into an 8-float block
            for (; j < N; j++) ptr_C[i * stride_C0 + j] += a_ik * ptr_B[k * stride_B0 + j];
        }
    }

    return C;    
}


// The ultimate version: Combines Cache Tiling + AVX2 SIMD optimizations + Autograd tracking
Tensor Tensor::matmul_tiled_simd(const Tensor& B, size_t tile_size) const {
    if (!this->is_contiguous() || !B.is_contiguous()) return this->contiguous().matmul_tiled_simd(B.contiguous(), tile_size);

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = B.shape_[1];

    bool req_grad = requires_grad_ || B.requires_grad_;
    Tensor C({M, N}, req_grad);
    
    float* ptr_A = storage_->data() + offset_;
    float* ptr_B = B.storage_->data() + B.offset_;
    float* ptr_C = C.storage_->data() + C.offset_;

    size_t stride_A0 = strides_[0], stride_A1 = strides_[1];
    size_t stride_B0 = B.strides_[0];
    size_t stride_C0 = C.strides_[0];

    // Block logic (Tiling)
    for (size_t i0 = 0; i0 < M; i0 += tile_size) {
        for (size_t k0 = 0; k0 < K; k0 += tile_size) {
            for (size_t j0 = 0; j0 < N; j0 += tile_size) {
                size_t i_max = std::min(i0 + tile_size, M);
                size_t k_max = std::min(k0 + tile_size, K);
                size_t j_max = std::min(j0 + tile_size, N);

                for (size_t i = i0; i < i_max; i++) {
                    for (size_t k = k0; k < k_max; k++) {
                        float a_ik = ptr_A[i * stride_A0 + k * stride_A1];
                        __m256 a_vec = _mm256_set1_ps(a_ik);
                        size_t j = j0;

                        // SIMD logic
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 c_vec = _mm256_loadu_ps(&ptr_C[i * stride_C0 + j]);
                            __m256 b_vec = _mm256_loadu_ps(&ptr_B[k * stride_B0 + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&ptr_C[i * stride_C0 + j], c_vec);
                        }
                        for (; j < j_max; j++) ptr_C[i * stride_C0 + j] += a_ik * ptr_B[k * stride_B0 + j];
                    }
                }
            }
        }
    }

    // Set up backpropagation for Matrix Multiplication (calculating dL/dA and dL/dB)
    if (req_grad) {
        Tensor A = *this;
        C.parents_ = {A, B};

        C.backward_op_ = [A, B, C, tile_size] () mutable {
            Tensor grad_C(C.grad_, C.shape(), C.strides(), C.offset(), false);
            
            // Grad of A = Grad of C * (B transposed)
            if (A.requires_grad_ && A.grad_) {
                Tensor B_T = B.transpose(0, 1);
                Tensor grad_A_update = grad_C.matmul_tiled_simd(B_T, tile_size);
                
                size_t size = A.grad_->size();
                for (size_t i = 0; i < size; i++) A.grad_->data()[i] += grad_A_update.storage()->data()[i];
            }

            // Grad of B = (A transposed) * Grad of C
            if (B.requires_grad_ && B.grad_) {
                Tensor A_T = A.transpose(0, 1);
                Tensor grad_B_update = A_T.matmul_tiled_simd(grad_C, tile_size);

                size_t size = B.grad_->size();
                for (size_t i = 0; i < size; i++) B.grad_->data()[i] += grad_B_update.storage()->data()[i];
            }
        };
    }

    return C;
}

// The Autograd Engine: Performs topological sort and calls backward functions
void Tensor::backward() {
    std::vector<Tensor> topo_order;
    std::set<void*> visited;

    // DFS to build graph order
    std::function<void(Tensor)> build_topo = [&](Tensor t) {
        // Create unique identifier for the tensor
        void* id = t.grad_ ? static_cast<void*>(t.grad_.get()) : static_cast<void*>(t.storage_.get());
        if (visited.find(id) == visited.end()) {
            visited.insert(id);
            for (auto& parent : t.parents_) { build_topo(parent); } // Visit parents first
            topo_order.push_back(t);
        }
    };

    build_topo(*this);
    
    // Seed the gradient of the loss output with 1.0 to start backprop
    if (grad_) for (size_t i = 0; i < grad_->size(); i++) grad_->data()[i] = 1.0f;

    // Reverse the topological order to calculate gradients backwards to the inputs/weights
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); it++) {
        if (it->backward_op_) it->backward_op_();
    }
}
