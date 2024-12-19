from data import load_data
from graph import plot_initial_data

# Load data
A_points, B_points = load_data()
plot_initial_data(A_points, B_points)