#include <iostream>
#include <iomanip>
#include "tensor.hpp"
#include "nn.hpp"
#include "optim.hpp"

int main() {
    std::cout << "=======================================\n";
    std::cout << "  XOR Problem Training via Custom MLP\n";
    std::cout << "=======================================\n";

    // 1. Prepare the XOR Dataset
    // Inputs: 4 samples, 2 features each. Set requires_grad to false since we don't update inputs.
    Tensor X({4, 2}, false);
    X.at({0, 0}) = 0.0f; X.at({0, 1}) = 0.0f;
    X.at({1, 0}) = 0.0f; X.at({1, 1}) = 1.0f;
    X.at({2, 0}) = 1.0f; X.at({2, 1}) = 0.0f;
    X.at({3, 0}) = 1.0f; X.at({3, 1}) = 1.0f;

    // Targets: Expected output for the XOR logic gate
    Tensor Y({4, 1}, false);
    Y.at({0, 0}) = 0.0f;
    Y.at({1, 0}) = 1.0f;
    Y.at({2, 0}) = 1.0f;
    Y.at({3, 0}) = 0.0f;

    // 2. Define the Neural Network Architecture
    // A simple Multi-Layer Perceptron (MLP) with 1 hidden layer
    nn::Linear layer1(2, 4);
    nn::Linear layer2(4, 1);

    // 3. Setup the Optimizer
    float learning_rate = 1.0f;
    // Pass the parameters (weights and biases) of both layers to the SGD optimizer
    SGD optimizer({&layer1.weight, &layer1.bias, &layer2.weight, &layer2.bias}, learning_rate);
    
    int epochs = 5000;

    // 4. Training loop
    for (int epoch = 1; epoch <= epochs; epoch++) {
        // Clear previous gradients
        optimizer.zero_grad();

        // Forward Pass: Layer 1 -> ReLU -> Layer 2 -> Sigmoid
        Tensor h1 = layer1.forward(X);
        Tensor h1_act = nn::functional::relu(h1);

        Tensor h2 = layer2.forward(h1_act);
        Tensor pred = nn::functional::sigmoid(h2);
        
        // Calculate Mean Squared Error Loss
        Tensor loss = nn::functional::mse_loss(pred, Y);
        
        // Backward Pass: Compute gradients using autograd
        loss.backward();
        // Update weights using the computed gradients
        optimizer.step();

        // Print loss every 1000 epochs
        if (epoch % 1000 == 0) {
            std::cout << "Epoch [" << std::setw(4) << epoch << "/" << epochs << "] "
                      << "Loss: " << std::fixed << std::setprecision(6)
                      << loss.storage()->data()[0] << "\n";
        }
    }

    std::cout << "\n Training Complete! Testing model Predictions:\n";

    // 5. Model Inference (Testing the trained model)
    Tensor h1 = layer1.forward(X);
    Tensor h1_act = nn::functional::relu(h1);
    Tensor h2 = layer2.forward(h1_act);
    Tensor pred = nn::functional::sigmoid(h2);

    // Print out the final predictions vs target values
    for (size_t i = 0; i < 4; i++) {
        float x1 = X.at({i, 0});
        float x2 = X.at({i, 1});
        float target = Y.at({i, 0});
        float prediction = pred.storage()->data()[i];
        
        std::cout << "Input: [" << x1 << ", " << x2 << "] "
                  << "=> Target: " << target
                  << "| Pred: " << std::fixed << std::setprecision(4) << prediction
                  << " (" << (prediction > 0.5f ? "1" : "0") << ")\n";
    }

    return 0;
}
