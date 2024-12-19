import matplotlib.pyplot as plt
import numpy as np

# Plotting initial points
def plot_initial_data(A, B):
    for a in A:
        plt.scatter(a[0], a[1], color='red')
    for b in B:
        plt.scatter(b[0], b[1], color='blue')
    
    # Saving the plot
    plt.savefig("initial_data.png")
    
    plt.show()
    
    return

# Plotting the final decision boundary
def plot_final_data(A, B, W):
    for a in A:
        plt.scatter(a[0], a[1], color='red')
    for b in B:
        plt.scatter(b[0], b[1], color='blue')
    
    x = np.linspace(0, 10, 100)
    y = (-W[2] - W[0] * x) / W[1]
    plt.plot(x, y, color='black')
    
    # Saving the plot
    plt.savefig("final_data.png")
    
    plt.show()
    
    return