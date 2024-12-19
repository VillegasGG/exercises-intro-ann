import numpy as np
from data import load_data
from graph import plot_initial_data, plot_final_data

# Load data
A_points, B_points = load_data()
plot_initial_data(A_points, B_points)

# Convert the data to numpy arrays
A = np.array(A_points)
B = np.array(B_points)

# Create data and labels
data = np.concatenate((A, B))
labels = np.array([1] * len(A) + [0] * len(B))

# Print data and labels
print("Data:", data)
print("Labels:", labels)

# Add bias to the data
data = np.hstack((data, np.ones((data.shape[0], 1))))

# Print data with bias
print("Data with bias:", data)

# Initialize weights
W = np.random.rand(3)
alpha = 0.1 # Learning rate

# Activation function
def activation_function(x):
    return 1 if x > 0 else 0

# Train the perceptron
for epoch in range(100):
    error_count = 0
    for j in range(data.shape[0]):
        z = np.dot(W, data[j])
        y_hat = activation_function(z)
        error = labels[j] - y_hat

        # Update weights
        if error != 0:
            W += alpha * error * data[j]
            error_count += 1
        
    if error_count == 0:
        break

# Show the final weights
print("Final weights:", W)

# Show number of iterations
print("Number of iterations:", epoch + 1)

# Plot the final decision boundary
plot_final_data(A_points, B_points, W)


