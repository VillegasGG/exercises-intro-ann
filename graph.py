import matplotlib.pyplot as plt

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

