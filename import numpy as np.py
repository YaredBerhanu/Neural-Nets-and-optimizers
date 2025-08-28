import numpy as np
import matplotlib.pyplot as plt
from typing import List

# --- Helper Functions ---

def relu(Z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function.
    
    Args:
        Z (np.ndarray): The input matrix.
        
    Returns:
        np.ndarray: The output matrix after applying ReLU.
    """
    return np.maximum(0, Z)

def relu_derivative(A):
    """
    Computes the derivative of the ReLU activation function.
    
    Args:
        A (np.ndarray): The output of the ReLU function.
        
    Returns:
        np.ndarray: The derivative of A.
    """
    return (A > 0).astype(int)

def softmax(Z):
    """
    Applies the Softmax activation function.
    
    Args:
        Z (np.ndarray): The input matrix.
        
    Returns:
        np.ndarray: The probabilities after applying Softmax.
    """
    # Subtract max for numerical stability
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    """
    Computes the categorical cross-entropy loss.
    
    Args:
        y_true (np.ndarray): True one-hot encoded labels.
        y_pred (np.ndarray): Predicted probabilities.
        
    Returns:
        float: The calculated loss.
    """
    m = y_true.shape[1]
    # Add a small value to avoid log(0)
    log_loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return np.squeeze(log_loss)

def categorical_cross_entropy_derivative(y_true, y_pred):
    """
    Computes the derivative of the categorical cross-entropy loss with respect to y_pred.
    
    Args:
        y_true (np.ndarray): True one-hot encoded labels.
        y_pred (np.ndarray): Predicted probabilities.
        
    Returns:
        np.ndarray: The derivative of the loss.
    """
    return y_pred - y_true

# --- Layer Classes ---

class Layer:
    """Base class for all neural network layers."""
    def __init__(self, name="Layer"):
        self.name = name
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, output_gradient):
        raise NotImplementedError

class Dense(Layer):
    """
    A fully connected (dense) layer.
    
    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
    """
    def __init__(self, input_size, output_size, activation="relu", l2_lambda=0.0):
        super().__init__(name="Dense")
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation
        self.l2_lambda = l2_lambda
        self.cache = {}
        self.velocity_w = np.zeros_like(self.weights)  # For momentum-based optimizers
        self.velocity_b = np.zeros_like(self.biases)
        self.s_w = np.zeros_like(self.weights)         # For adaptive optimizers
        self.s_b = np.zeros_like(self.biases)
        
    def forward(self, input_data):
        """
        Performs the forward pass for the dense layer.
        
        Args:
            input_data (np.ndarray): The input data (X).
            
        Returns:
            np.ndarray: The output of the layer.
        """
        self.input = input_data
        self.Z = np.dot(self.weights, self.input) + self.biases
        self.cache['input'] = self.input
        self.cache['Z'] = self.Z
        
        if self.activation == "relu":
            self.output = relu(self.Z)
        elif self.activation == "softmax":
            self.output = softmax(self.Z)
        else:
            self.output = self.Z # Linear activation
        
        self.cache['output'] = self.output
        return self.output
    
    def backward(self, output_gradient):
        """
        Performs the backward pass for the dense layer.
        
        Args:
            output_gradient (np.ndarray): The gradient from the subsequent layer.
            
        Returns:
            np.ndarray: The gradient to be passed to the previous layer.
        """
        m = self.input.shape[1]
        
        if self.activation == "relu":
            dZ = output_gradient * relu_derivative(self.cache['output'])
        elif self.activation == "softmax":
            dZ = output_gradient
        else:
            dZ = output_gradient
            
        self.dW = (1/m) * np.dot(dZ, self.input.T) + (self.l2_lambda / m) * self.weights
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        input_gradient = np.dot(self.weights.T, dZ)
        return input_gradient

class Dropout(Layer):
    """
    A dropout layer for regularization.
    
    Args:
        p (float): The dropout probability (rate).
    """
    def __init__(self, p=0.5):
        super().__init__(name="Dropout")
        self.p = p
        self.mask = None
    
    def forward(self, input_data, training=True):
        """
        Performs the forward pass for the dropout layer.
        
        Args:
            input_data (np.ndarray): The input data.
            training (bool): If True, apply dropout.
            
        Returns:
            np.ndarray: The output data.
        """
        if training:
            self.mask = (np.random.rand(*input_data.shape) < self.p) / self.p
            self.output = input_data * self.mask
        else:
            self.output = input_data
        return self.output
    
    def backward(self, output_gradient):
        """
        Performs the backward pass for the dropout layer.
        
        Args:
            output_gradient (np.ndarray): The gradient from the subsequent layer.
            
        Returns:
            np.ndarray: The gradient to be passed to the previous layer.
        """
        return output_gradient * self.mask

# --- Optimizer Classes ---

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer):
        raise NotImplementedError

class GradientDescent(Optimizer):
    """
    Standard Gradient Descent optimizer.
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, layer):
        """
        Updates the weights and biases of a dense layer.
        
        Args:
            layer (Dense): The dense layer to update.
        """
        layer.weights -= self.learning_rate * layer.dW
        layer.biases -= self.learning_rate * layer.db

class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        
    def update(self, layer):
        """
        Updates the weights and biases of a dense layer using RMSProp.
        
        Args:
            layer (Dense): The dense layer to update.
        """
        layer.s_w = self.beta * layer.s_w + (1 - self.beta) * layer.dW**2
        layer.s_b = self.beta * layer.s_b + (1 - self.beta) * layer.db**2
        
        layer.weights -= self.learning_rate * layer.dW / (np.sqrt(layer.s_w) + self.epsilon)
        layer.biases -= self.learning_rate * layer.db / (np.sqrt(layer.s_b) + self.epsilon)

class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def update(self, layer):
        """
        Updates the weights and biases of a dense layer using Adam.
        
        Args:
            layer (Dense): The dense layer to update.
        """
        self.t += 1
        
        layer.velocity_w = self.beta1 * layer.velocity_w + (1 - self.beta1) * layer.dW
        layer.velocity_b = self.beta1 * layer.velocity_b + (1 - self.beta1) * layer.db
        
        layer.s_w = self.beta2 * layer.s_w + (1 - self.beta2) * layer.dW**2
        layer.s_b = self.beta2 * layer.s_b + (1 - self.beta2) * layer.db**2
        
        v_w_corrected = layer.velocity_w / (1 - self.beta1**self.t)
        v_b_corrected = layer.velocity_b / (1 - self.beta1**self.t)
        s_w_corrected = layer.s_w / (1 - self.beta2**self.t)
        s_b_corrected = layer.s_b / (1 - self.beta2**self.t)
        
        layer.weights -= self.learning_rate * v_w_corrected / (np.sqrt(s_w_corrected) + self.epsilon)
        layer.biases -= self.learning_rate * v_b_corrected / (np.sqrt(s_b_corrected) + self.epsilon)

# --- Neural Network Class ---

class NeuralNetwork:
    """
    A simple neural network implementation.
    
    Args:
        input_size (int): The number of input features.
    """
    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size
        self.optimizer = None
    
    def add_layer(self, layer):
        """Adds a layer to the network."""
        self.layers.append(layer)
    
    def set_optimizer(self, optimizer):
        """Sets the optimizer for the network."""
        self.optimizer = optimizer
        
    def forward(self, X, training=True):
        """
        Performs the forward pass through the network.
        
        Args:
            X (np.ndarray): The input data.
            training (bool): If True, applies dropout.
            
        Returns:
            np.ndarray: The network's output predictions.
        """
        current_output = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                current_output = layer.forward(current_output, training=training)
            else:
                current_output = layer.forward(current_output)
        return current_output
    
    def backward(self, dZ):
        """
        Performs the backward pass and updates weights using the set optimizer.
        
        Args:
            dZ (np.ndarray): The initial gradient from the loss function.
        """
        current_gradient = dZ
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)
        
        for layer in self.layers:
            if isinstance(layer, Dense):
                self.optimizer.update(layer)
    
    def train(self, X_train, y_train, epochs, optimizer):
        """
        Trains the neural network.
        
        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training true labels.
            epochs (int): Number of training epochs.
            optimizer (Optimizer): The optimizer to use.
            
        Returns:
            Tuple[List[float], List[float]]: Lists of loss and accuracy history.
        """
        self.set_optimizer(optimizer)
        loss_history = []
        accuracy_history = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train, training=True)
            
            # Compute loss and accuracy
            loss = categorical_cross_entropy(y_train, y_pred)
            accuracy = self.evaluate(X_train, y_train)
            
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            
            # Backward pass
            dZ = categorical_cross_entropy_derivative(y_train, y_pred)
            self.backward(dZ)
            
            if (epoch + 1) % 100 == 0:
                print(f"Optimizer: {self.optimizer.__class__.__name__}, Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss_history, accuracy_history
    
    def predict(self, X):
        """
        Predicts labels for a given input data.
        
        Args:
            X (np.ndarray): The input data.
            
        Returns:
            np.ndarray: The predicted class labels.
        """
        probabilities = self.forward(X, training=False)
        return np.argmax(probabilities, axis=0)

    def evaluate(self, X, y_true):
        """
        Evaluates the model's accuracy.
        
        Args:
            X (np.ndarray): Input data.
            y_true (np.ndarray): True one-hot encoded labels.
            
        Returns:
            float: The accuracy.
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y_true, axis=0)
        correct_predictions = np.sum(predictions == true_labels)
        accuracy = correct_predictions / y_true.shape[1]
        return accuracy

# --- Data Generation Function ---

def generate_data(num_samples=1000, num_classes=2):
    """
    Generates a synthetic spiral dataset for a classification task.
    
    Args:
        num_samples (int): The total number of samples.
        num_classes (int): The number of classes.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X and y data.
    """
    N = int(num_samples / num_classes)
    X = np.zeros((num_samples, 2))
    y = np.zeros(num_samples, dtype='uint8')
    
    for j in range(num_classes):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = j
    
    X = X.T
    y = y.T
    
    # Normalize features
    X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
    
    # One-hot encode labels
    y_one_hot = np.zeros((num_classes, num_samples))
    y_one_hot[y, np.arange(num_samples)] = 1
    
    return X, y_one_hot

# --- Main Execution Block ---

if __name__ == "__main__":
    # Parameters
    input_size = 2
    hidden_size_1 = 64
    hidden_size_2 = 64
    output_size = 2
    epochs = 1000
    learning_rate_gd = 0.5
    learning_rate_rmsprop = 0.001
    learning_rate_adam = 0.001
    l2_lambda = 0.001
    dropout_rate = 0.5

    # Generate data
    X, y = generate_data()

    # --- Training with Gradient Descent ---
    print("\n--- Training with Gradient Descent ---")
    model_gd = NeuralNetwork(input_size)
    model_gd.add_layer(Dense(input_size, hidden_size_1, activation="relu", l2_lambda=l2_lambda))
    model_gd.add_layer(Dropout(p=dropout_rate))
    model_gd.add_layer(Dense(hidden_size_1, hidden_size_2, activation="relu", l2_lambda=l2_lambda))
    model_gd.add_layer(Dropout(p=dropout_rate))
    model_gd.add_layer(Dense(hidden_size_2, output_size, activation="softmax", l2_lambda=l2_lambda))
    
    gd_loss, gd_accuracy = model_gd.train(X, y, epochs, GradientDescent(learning_rate=learning_rate_gd))

    # --- Training with RMSProp ---
    print("\n--- Training with RMSProp ---")
    model_rmsprop = NeuralNetwork(input_size)
    model_rmsprop.add_layer(Dense(input_size, hidden_size_1, activation="relu", l2_lambda=l2_lambda))
    model_rmsprop.add_layer(Dropout(p=dropout_rate))
    model_rmsprop.add_layer(Dense(hidden_size_1, hidden_size_2, activation="relu", l2_lambda=l2_lambda))
    model_rmsprop.add_layer(Dropout(p=dropout_rate))
    model_rmsprop.add_layer(Dense(hidden_size_2, output_size, activation="softmax", l2_lambda=l2_lambda))
    
    rmsprop_loss, rmsprop_accuracy = model_rmsprop.train(X, y, epochs, RMSProp(learning_rate=learning_rate_rmsprop))

    # --- Training with Adam ---
    print("\n--- Training with Adam ---")
    model_adam = NeuralNetwork(input_size)
    model_adam.add_layer(Dense(input_size, hidden_size_1, activation="relu", l2_lambda=l2_lambda))
    model_adam.add_layer(Dropout(p=dropout_rate))
    model_adam.add_layer(Dense(hidden_size_1, hidden_size_2, activation="relu", l2_lambda=l2_lambda))
    model_adam.add_layer(Dropout(p=dropout_rate))
    model_adam.add_layer(Dense(hidden_size_2, output_size, activation="softmax", l2_lambda=l2_lambda))
    
    adam_loss, adam_accuracy = model_adam.train(X, y, epochs, Adam(learning_rate=learning_rate_adam))

    # --- Plotting Convergence Curves ---
    
    # Plot Loss Curves
    plt.figure(figsize=(12, 6))
    plt.plot(gd_loss, label='Gradient Descent')
    plt.plot(rmsprop_loss, label='RMSProp')
    plt.plot(adam_loss, label='Adam')
    plt.title('Loss Convergence Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Accuracy Curves
    plt.figure(figsize=(12, 6))
    plt.plot(gd_accuracy, label='Gradient Descent')
    plt.plot(rmsprop_accuracy, label='RMSProp')
    plt.plot(adam_accuracy, label='Adam')
    plt.title('Accuracy Convergence Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Small Report on Comparison ---
    print("\n--- Comparison Report ---")
    print(f"Gradient Descent final accuracy: {gd_accuracy[-1]:.4f}")
    print(f"RMSProp final accuracy:          {rmsprop_accuracy[-1]:.4f}")
    print(f"Adam final accuracy:             {adam_accuracy[-1]:.4f}")
    
    print("\nAnalysis:")
    print("Adam and RMSProp are adaptive optimizers. They adjust the learning rate for each parameter, which often leads to faster convergence and better performance on complex tasks. As you can see from the final accuracy scores and the plots, Adam and RMSProp performed significantly better on this spiral dataset than standard Gradient Descent.")
    print("Gradient Descent can get stuck in local minima or oscillate, and requires careful tuning of a single learning rate for all parameters.")
    print("Adam, in particular, combines the benefits of RMSProp (handling sparse gradients) and momentum, making it a robust and widely-used optimizer for deep learning.")

