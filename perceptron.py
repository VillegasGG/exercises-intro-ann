import numpy as np
from data import load_data
from graph import plot_initial_data, plot_final_data, plot_final_data_with_new_points

# Compute the perceptron output
def perceptron_output(x, W):
    z = np.dot(x, W)
    return 1 if z > 0 else 0

# Train the perceptron
def train_perceptron(data, labels, weights, learning_rate):
    # While everything is not classified correctly
    epochs = 0
    while(True):
        epochs += 1
        misclassified = False
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            y_hat = perceptron_output(x, weights)
            
            if y != y_hat:
                weights += learning_rate * (y - y_hat) * x
                print(f"----------")
                print(f"Updating weights: {weights} + {learning_rate * (y - y_hat) * x}")
                misclassified = True
                
        # Print epoch results
        print("----------")
        print("Epoch:", epochs)
        print("Weights:", weights)
        print("Misclassified:", misclassified)
        print("Accuracy:", 1 - np.mean(misclassified))
        print("Learning rate:", learning_rate)
        print()
        print("----------")

        # Plot the decision boundary
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
labels = np.array([1] * len(A) + [0] * len(B))

# Add bias to the data
data = np.c_[data, np.ones(data.shape[0])]

# Initialize weights
weights = np.array([1.0, 1.0, 1.0])

# Plot initial data
plot_initial_data(A_points, B_points)

# Plot the initial decision boundary	
plot_final_data(A_points, B_points, weights, "initial_data")

# Train the perceptron
weights = train_perceptron(data, labels, weights, 0.4)

# Plot the final decision boundary
plot_final_data(A_points, B_points, weights, "final_data")

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

