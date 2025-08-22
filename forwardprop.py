import numpy as np

from backprop import backward_propagation

# Activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Initialize weights and biases with He initialization
def initialize_parameters(input_size, hidden1_size, hidden2_size, hidden3_size, output_size, seed=42):
    np.random.seed(seed)
    W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / hidden1_size)
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, hidden3_size) * np.sqrt(2 / hidden2_size)
    b3 = np.zeros((1, hidden3_size))
    W4 = np.random.randn(hidden3_size, output_size) * np.sqrt(2 / hidden3_size)
    b4 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3, W4, b4

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = softmax(Z4)
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4


# Training loop
def train(X_train, y_train, W1, b1, W2, b2, W3, b3, W4, b4, epochs, learning_rate, lambda_reg=0.001):
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_propagation(X_train, W1, b1, W2, b2, W3, b3, W4, b4)
        W1, b1, W2, b2, W3, b3, W4, b4, loss = backward_propagation(
            X_train, y_train, Z1, A1, Z2, A2, Z3, A3, Z4, A4,
            W1, b1, W2, b2, W3, b3, W4, b4, learning_rate, lambda_reg
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1, b1, W2, b2, W3, b3, W4, b4