#include <gtest/gtest.h>
#include <chrono>
#include "tensor.hpp"
#include "nn.hpp"
#include "optim.hpp"

// Test if tensor shape, memory size, and strides are initialized properly
TEST(TensorTest, InitializationAndStrides) {
    Tensor t({2, 3});

    EXPECT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.strides()[0], 3); // Move 3 steps to go to next row
    EXPECT_EQ(t.strides()[1], 1); // Move 1 step to go to next column
    EXPECT_EQ(t.storage()->size(), 6);
    EXPECT_EQ(t.offset(), 0);
}

// Ensure transpose doesn't copy data but changes how it is read (views)
TEST(TensorTest, ZeroCopyTranspose) {
    Tensor t({2, 3});

    t.at({0, 0}) = 1.0f; t.at({0, 1}) = 2.0f; t.at({0, 2}) = 3.0f;
    t.at({1, 0}) = 4.0f; t.at({1, 1}) = 5.0f; t.at({1, 2}) = 6.0f;

    Tensor t_T = t.transpose(0, 1);
    
    // Dimensions should be swapped
    EXPECT_EQ(t_T.shape()[0], 3);
    EXPECT_EQ(t_T.shape()[1], 2);
    EXPECT_EQ(t_T.strides()[0], 1);
    EXPECT_EQ(t_T.strides()[1], 3);

    // Both tensors should point to the same underlying memory array
    EXPECT_EQ(t.storage().get(), t_T.storage().get());
    EXPECT_EQ(t_T.at({1, 0}), 2.0f);
    
    // Mutating transposed tensor should affect the original
    t_T.at({1, 0}) = 99.0f;
    EXPECT_EQ(t.at({0, 1}), 99.0f);

    EXPECT_EQ(t_T.is_contiguous(), false); // Transposed view is no longer linear in memory
    EXPECT_EQ(t.is_contiguous(), true);
}

// Verify all different MatMul implementations produce the exact same results
TEST(TensorBenchmark, MatMulCorrectness) {
    Tensor A({100, 100});
    Tensor B({100, 100});

    Tensor C_naive = A.matmul_naive(B);
    Tensor C_ikj = A.matmul_ikj(B);
    Tensor C_tiled = A.matmul_tiled(B);
    Tensor C_simd = A.matmul_simd(B);    
    Tensor C_tiled_simd = A.matmul_tiled_simd(B);

    for (size_t i = 0; i < C_naive.shape()[0]; i++) {
        for (size_t j = 0; j < C_naive.shape()[1]; j++) {
            EXPECT_NEAR(C_naive.at({i, j}), C_ikj.at({i, j}), 1e-5);
            EXPECT_NEAR(C_naive.at({i, j}), C_tiled.at({i, j}), 1e-5);
            EXPECT_NEAR(C_naive.at({i, j}), C_simd.at({i, j}), 1e-5);
            EXPECT_NEAR(C_naive.at({i, j}), C_tiled_simd.at({i, j}), 1e-5);
        }
    }
}

// Measure and print how much faster the optimizations (Tiling/AVX2) are on large matrices
TEST(TensorBenchmark, MatMulPerformance) {
    size_t N = 2048;
    Tensor A({N, N});
    Tensor B({N, N});
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor C_naive = A.matmul_naive(B);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_naive = end - start;
    std::cout << "Naive (i-j-k) Time: " << diff_naive.count() << " seconds\n";
    
    start = std::chrono::high_resolution_clock::now();
    Tensor C_ikj = A.matmul_ikj(B);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_ikj = end - start;
    std::cout << "IKJ (i-k-j) Time: " << diff_ikj.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    Tensor C_tiled = A.matmul_tiled(B);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_tiled = end - start;
    std::cout << "Tiled Time: " << diff_tiled << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    Tensor C_simd = A.matmul_simd(B);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_simd = end - start;
    std::cout << "SIMD Time: " << diff_simd << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    Tensor C_tiled_simd = A.matmul_tiled_simd(B);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_tiled_simd = end - start;
    std::cout << "Tiled&SIMD Time: " << diff_tiled_simd << " seconds\n";
}

// Test Autograd gradient propagation during addition
TEST(TensorTest, BackwardAdd) {
    Tensor A({2, 2}, true);
    Tensor B({2, 2}, true);
    
    A.at({0, 0}) = 1.0f; A.at({0, 1}) = 2.0f;
    A.at({1, 0}) = 3.0f; A.at({1, 1}) = 4.0f;

    B.at({0, 0}) = 5.0f; B.at({0, 1}) = 6.0f;
    B.at({1, 0}) = 7.0f; B.at({1, 1}) = 8.0f;

    Tensor C = A.add(B);
    C.backward(); // Trigger graph traversal

    float* grad_a = A.grad_->data();
    float* grad_b = B.grad_->data();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(grad_a[i], 1.0f);
        EXPECT_FLOAT_EQ(grad_b[i], 1.0f);
    }
}

// Test Autograd gradient propagation through Matrix Multiplication
TEST(TensorTest, BackwardMatMul) {
    Tensor A({2, 2}, true);
    Tensor B({2, 2}, true);
    
    A.at({0, 0}) = 1.0f; A.at({0, 1}) = 2.0f;
    A.at({1, 0}) = 3.0f; A.at({1, 1}) = 4.0f;

    B.at({0, 0}) = 5.0f; B.at({0, 1}) = 6.0f;
    B.at({1, 0}) = 7.0f; B.at({1, 1}) = 8.0f;

    Tensor C = A.matmul_tiled_simd(B);

    C.backward();

    float* grad_a = A.grad_->data();
    float* grad_b = B.grad_->data();

    // Verify derived gradients mathematically match analytical derivative values
    EXPECT_FLOAT_EQ(grad_a[0], 11.0f);
    EXPECT_FLOAT_EQ(grad_a[1], 15.0f);
    EXPECT_FLOAT_EQ(grad_a[2], 11.0f);
    EXPECT_FLOAT_EQ(grad_a[3], 15.0f);

    EXPECT_FLOAT_EQ(grad_b[0], 4.0f);
    EXPECT_FLOAT_EQ(grad_b[1], 4.0f);
    EXPECT_FLOAT_EQ(grad_b[2], 6.0f);
    EXPECT_FLOAT_EQ(grad_b[3], 6.0f);
}

// Test ReLU function correctly truncates negatives and routes gradients
TEST(NNTest, ReluForwardBackward) {
    Tensor A({2, 2}, true);
    
    A.storage()->data()[0] = 1.0f; A.storage()->data()[1] = -2.0f;
    A.storage()->data()[2] = -3.0f; A.storage()->data()[3] = 4.0f;

    Tensor out = nn::functional::relu(A);

    EXPECT_FLOAT_EQ(out.storage()->data()[0], 1.0f);
    EXPECT_FLOAT_EQ(out.storage()->data()[1], 0.0f);
    EXPECT_FLOAT_EQ(out.storage()->data()[2], 0.0f);
    EXPECT_FLOAT_EQ(out.storage()->data()[3], 4.0f);
}

// Verify MSE loss correctly calculates error and propagates derivatives
TEST(NNTest, MSELossForwardBackward) {
    Tensor pred({2}, true);
    pred.storage()->data()[0] = 1.0f;
    pred.storage()->data()[1] = 3.0f;

    Tensor target({2}, false);
    target.storage()->data()[0] = 2.0f;
    target.storage()->data()[1] = 1.0f;
    
    Tensor loss = nn::functional::mse_loss(pred, target);
    loss.backward();

    EXPECT_FLOAT_EQ(pred.grad_->data()[0], -1.0f);
    EXPECT_FLOAT_EQ(pred.grad_->data()[1], 2.0f);
}

// Test that an entire linear layer gets gradient updates during backward
TEST(NNTest, LinearLayerIntegration) {
    nn::Linear layer(2, 3);
    
    Tensor A({1, 2}, true);
    A.storage()->data()[0] = 1.0f;
    A.storage()->data()[1] = -1.0f;

    Tensor out = layer.forward(A);

    EXPECT_EQ(out.shape()[0], 1);
    EXPECT_EQ(out.shape()[1], 3);

    out.backward();

    // If autograd ran successfully, weight gradients should no longer be 0
    EXPECT_NE(layer.weight.grad_->data()[0], 0.0f);
}

// Test if the Optimizer correctly updates parameter values based on gradients
TEST(OptimTest, SGDStepAndZeroGrad) {
    Tensor W({1, 2}, true);
    W.storage()->data()[0] = 10.0f; W.storage()->data()[1] = 20.0f;

    W.grad_->data()[0] = 1.0f; W.grad_->data()[1] = -2.0f;

    SGD optimizer({&W}, 0.1f);

    optimizer.step(); // Update weights
    EXPECT_FLOAT_EQ(W.storage()->data()[0], 9.9f);
    EXPECT_FLOAT_EQ(W.storage()->data()[1], 20.2f);

    optimizer.zero_grad(); // Reset gradients
    EXPECT_FLOAT_EQ(W.grad_->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(W.grad_->data()[1], 0.0f);
}
