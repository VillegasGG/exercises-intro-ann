"""
Perceptron con función de activación sigmoidal
y con bias mediante la regla delta adaptada
"""

import numpy as np
from data import load_data
from graph import plot_initial_data, plot_final_data, plot_final_data_with_new_points

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron output
def perceptron_output(x, weights):
    return sigmoid(np.dot(x, weights))

# Train the perceptron
def train_perceptron(data, labels, weights, learning_rate):
    epochs = 0
    while(True):
        epochs += 1
        misclassified = False
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            y_hat = perceptron_output(x, weights)

            # Round classification
            if y_hat >= 0.5:
                y_hat_aux = 1
            else:
                y_hat_aux = -1

            error = y - y_hat
            weights += learning_rate * error * sigmoid_derivative(y_hat) * x
            
            # Verufy classification
            if y_hat_aux != y:
                misclassified = True
        
        # Print epoch results
        print("----------")
        print("Epoch:", epochs)
        print("Weights:", weights)
        print("Misclassified:", misclassified)
        print("Error:", error)
        print("----------")

        # Plot the decision boundary every 10 epochs
        if epochs % 10 == 0:
            plot_final_data(A, B, weights, "epoch_" + str(epochs))

        if not misclassified:
            break
    
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

# print the data and labels
print("Data:", data)
print("Labels:", labels)

# Add bias to the data
data = np.c_[data, np.ones(data.shape[0])]
original_data = np.copy(data)

# Scale the data (both x and y) to be between 0 and 1
data[:, 0] = data[:, 0] / np.max(data[:, 0])
data[:, 1] = data[:, 1] / np.max(data[:, 1])

# Obtain new A and B points for graphing
A = data[np.where(labels == 1)]
B = data[np.where(labels == -1)]

# Initialize weights
weights = np.array([0.5, -0.5, 1.0])

# Plot initial data
plot_initial_data(A_points, B_points)

# Plot the initial decision boundary	
plot_final_data(A_points, B_points, weights, "initial_data_sigmoid")

# Train the perceptron
learning_rate = 0.01
weights = train_perceptron(data, labels, weights, learning_rate)

# Rescale the weights
weights[0] = weights[0] / np.max(original_data[:, 0])
weights[1] = weights[1] / np.max(original_data[:, 1])
weights[2] = weights[2] / np.max(original_data[:, 2])

# Plot the final decision boundary
plot_final_data(A_points, B_points, weights, "final_data_sigmoid")

# Try to classify a new point

new_point = np.array([5, 5])
new_point = np.append(new_point, 1)
print("New point:", new_point)
print("Classification:", perceptron_output(new_point, weights))
plot_final_data_with_new_points(A_points, B_points, weights, [new_point], "final_data_with_new_point1")


new_point = np.array([6, 8])
new_point = np.append(new_point, 1)
print("New point:", new_point)
print("Classification:", perceptron_output(new_point, weights))
plot_final_data_with_new_points(A_points, B_points, weights, [new_point], "final_data_with_new_point2")
