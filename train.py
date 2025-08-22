import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from forwardprop import forward_propagation, initialize_parameters, train

# Load and prepare data
digits = load_digits() # This is a copy of the test set of the UCI ML hand-written digits datasets
X = digits.data
y = digits.target

print(f"feature shape: {X.shape}")
print(f"target shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define network sizes
input_size = X.shape[1]  # 64 features
hidden1_size = 128
hidden2_size = 64
hidden3_size = 32
output_size = 10  # 10 classes

# Initialize parameters

W1, b1, W2, b2, W3, b3, W4, b4 = initialize_parameters(
    input_size, hidden1_size, hidden2_size, hidden3_size, output_size
)

# Train the model
W1, b1, W2, b2, W3, b3, W4, b4 = train(
    X_train, y_train, W1, b1, W2, b2, W3, b3, W4, b4,
    epochs=100, learning_rate=0.1, lambda_reg=0.01
)

# Prediction function
def predict(X, W1, b1, W2, b2, W3, b3, W4, b4):
    _, _, _, _, _, _, _, A4 = forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4)
    return np.argmax(A4, axis=1)


# Make predictions on test set
y_pred = predict(X_test, W1, b1, W2, b2, W3, b3, W4, b4)

# Compute accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")