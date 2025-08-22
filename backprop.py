import numpy as np

""" 
Highlight: This is Vanilla Batch Gradient Descent
In this implementation, we update the parameters using the basic gradient descent rule:
parameter = parameter - learning_rate * gradient
This is applied to the entire training set (batch) in each epoch, making it Batch Gradient Descent (GD).
How it works: The gradients (dW, db) are computed as the average over all training examples, then scaled by a fixed learning rate.
To arrive at this: Start from the loss function, compute partial derivatives w.r.t. each parameter using chain rule (backprop), then apply the update rule.
Strengths: Simple, stable updates since it uses the full dataset.
Weaknesses: Slow for large datasets (computes full pass each time), can converge slowly or get stuck in plateaus/saddle points.
For teaching optimizers: This is the baseline. To improve, you could:
- Use Mini-batch GD: Split data into batches, update per batch for faster, noisier updates (enables stochasticity for better generalization).
- Add Momentum: Introduce velocity term to accelerate in consistent directions.
- Use Adaptive methods like Adam: Adapt lr per parameter based on first/second moments of gradients for faster convergence on sparse/high-dim data.
"""

# Backward propagation with L2 regularization
def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, Z4, A4,
                         W1, b1, W2, b2, W3, b3, W4, b4, learning_rate, lambda_reg):
    m = X.shape[0]

    # Cross-entropy loss
    cross_entropy_loss = -np.sum(np.log(A4[np.arange(m), y] + 1e-15)) / m

    # L2 regularization term
    reg_term = (lambda_reg / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    loss = cross_entropy_loss + reg_term

    # Output layer gradients
    dZ4 = A4.copy()
    dZ4[np.arange(m), y] -= 1
    dW4 = np.dot(A3.T, dZ4) / m + (lambda_reg / m) * W4
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    # Third hidden layer gradients
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * (Z3 > 0)
    dW3 = np.dot(A2.T, dZ3) / m + (lambda_reg / m) * W3
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # Second hidden layer gradients
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * (Z2 > 0)
    dW2 = np.dot(A1.T, dZ2) / m + (lambda_reg / m) * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # First hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m + (lambda_reg / m) * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    return W1, b1, W2, b2, W3, b3, W4, b4, loss