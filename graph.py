import matplotlib.pyplot as plt
import numpy as np

# Plotting initial points
def plot_initial_data(A, B):
    for a in A:
        plt.scatter(a[0], a[1], color='red')
    plt.scatter([], [], color='red', label='A')
    for b in B:
        plt.scatter(b[0], b[1], color='blue')
    plt.scatter([], [], color='blue', label='B')
    
    # Adding legend
    plt.legend()
    
    # Saving the plot
    plt.savefig('results/' + "initial_data.png")
    
    plt.close()
    
    return

# Plotting the final decision boundary
def plot_final_data(A, B, W, plot_name):
    for a in A:
        plt.scatter(a[0], a[1], color='red')
    plt.scatter([], [], color='red', label='A')
    for b in B:
        plt.scatter(b[0], b[1], color='blue')
    plt.scatter([], [], color='blue', label='B')

    x = np.linspace(0, 10, 100)
    y = (-W[2] - W[0] * x) / W[1]
    plt.plot(x, y, color='black')
    plt.legend()
    # Saving the plot
    plt.savefig('results/' + plot_name + ".png")
    
    plt.close()
    
    return

# Plotting the final decision boundary with new points
def plot_final_data_with_new_points(A, B, W, new_points, plot_name):
    for a in A:
        plt.scatter(a[0], a[1], color='red')
    plt.scatter([], [], color='red', label='A')
    for b in B:
        plt.scatter(b[0], b[1], color='blue')
    plt.scatter([], [], color='blue', label='B')
    
    for point in new_points:
        plt.scatter(point[0], point[1], color='green', label='New Point')
    
    x = np.linspace(0, 10, 100)
    y = (-W[2] - W[0] * x) / W[1]
    plt.plot(x, y, color='black')
    plt.legend()
    
    # Saving the plot
    plt.savefig('results/' + plot_name + ".png")
    
    plt.close()
    
    return