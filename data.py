import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data():
    print("Enter the coordinates of the A points in format x1 y1, x2 y2, ...")
    A = input('A: ')
    print("Enter the coordinates of the B points in format x1 y1, x2 y2, ...")
    B = input('B: ')

    A = A.split(',')
    B = B.split(',')

    A_points = parse_coordinates(A)
    B_points = parse_coordinates(B)
    
    print("A:", A_points)
    print("B:", B_points)

    return A_points, B_points

def parse_coordinates(coordinates):
    points = []
    for coordinate in coordinates:
        x, y = coordinate.strip().split(' ')
        x = int(x)
        y = int(y)
        points.append([x, y])
    return points

load_data()

# 2 6, 4 4, 6 3
# 4 10, 7 10, 9 8
