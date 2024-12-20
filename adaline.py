"""
Using the Adaline algorithm with bias and delta rule to
classify two classes of points.
"""
import numpy as np
from data import load_data
from graph import plot_initial_data, plot_final_data, plot_final_data_with_new_points

# Compute Adaline output
def adaline_output(data, weights):
    return np.dot(data, weights)

# Train the Adaline
def train_adaline(data, labels, weights, alpha):
    epochs = 100
    previous_error = 0
    for epoch in range(epochs):  
        total_error = 0
        for j in range(len(data)):
            z = adaline_output(data[j], weights)
            error = labels[j] - z
            weights += alpha * error * data[j]
            total_error += error ** 2 
        
        # Print epoch results
        print("----------")
        print("Epoch:", epoch)
        print("Weights:", weights)
        print("Error:", error)
        print("Total Error:", total_error)
        print("----------")

        # Early stopping
        if abs(previous_error - total_error) < 1e-5:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        previous_error = total_error

        # Plot the decision boundary every 10 epochs
        if epoch % 10 == 0:
            plot_final_data(A_points, B_points, weights, "epoch_" + str(epoch))
    
    return weights

# Load data
A_points, B_points = load_data()
plot_initial_data(A_points, B_points)

# Convert the data to numpy arrays
A = np.array(A_points)
B = np.array(B_points)

# Create data and labels
data = np.concatenate((A, B))
labels = np.array([1] * len(A) + [-1] * len(B))

# Add bias to the data
bias = np.ones((data.shape[0], 1))
data = np.c_[data, bias]

# Initialize weights
weights = np.array([1.0, 1.0, 5.0])

# Plot the initial decision boundary
plot_final_data(A_points, B_points, weights, "initial_data_adaline")

# Train the Adaline
alpha = 0.01
weights = train_adaline(data, labels, weights, alpha)

# Plot the final decision boundary
plot_final_data(A_points, B_points, weights, "final_data_adaline")

