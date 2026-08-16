# CoreNN_CPP: A Personal Deep Learning Sandbox

This is a personal practice project I built to understand how neural networks actually work under the hood using C++20. Instead of relying on PyTorch or TensorFlow, I wanted to try implementing the core components—like tensors, backpropagation (autograd), and matrix multiplication—from scratch.

The main goal here was learning, so the current setup focuses on a simple Multi-Layer Perceptron (MLP) that learns to solve the classic XOR problem.

## 🛠️ What I Built & Learned

*   **Custom Autograd Engine:** I implemented a basic computation graph. Tensors keep track of their parents and operations, allowing for automatic gradient calculation (`backward()`) using topological sorting.
*   **Zero-Copy Transpose:** Learned how to transpose matrices by just swapping strides instead of copying the underlying memory.
*   **Matrix Multiplication (MatMul) Optimization:** I experimented with different ways to speed up matrix multiplication. The code includes a naive approach, loop reordering (IKJ), memory tiling for cache locality, and finally, hardware acceleration using AVX2 SIMD instructions.
*   **Basic NN Modules:** Wrote simple `Linear` layers, `ReLU` / `Sigmoid` activations, and a Mean Squared Error (`MSELoss`) function.
*   **Optimizer:** Added a basic Stochastic Gradient Descent (`SGD`) optimizer to update the weights.

## 📂 Project Structure

*   `include/` & `src/`: Contains the core logic (`tensor`, `nn`, `optim`, `storage`).
*   `tests/test_corenn.cpp`: GoogleTest suite where I verify my math, test the autograd gradients, and benchmark the different MatMul implementations.
*   `train.cpp`: A simple playground script where I train an MLP on the XOR dataset.

## 🚀 How to Run It

You'll need a C++20 compiler and CMake. Also, a CPU that supports AVX2 instructions is recommended to run the SIMD optimized code.

```bash
mkdir build && cd build
cmake ..
make -j4

# Run the XOR training example
./train_benchmark

# Run the tests and MatMul performance benchmarks
./corenn_test
