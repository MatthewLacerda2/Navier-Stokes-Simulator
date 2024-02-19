import numpy as np
import matplotlib.pyplot as plt

# Define global variables
Lx = 1.0  # Domain size in x
Ly = 1.0  # Domain size in y
nx = 50   # Number of grid points in x
ny = 50   # Number of grid points in y
dx = Lx / (nx - 1)  # Grid spacing in x
dy = Ly / (ny - 1)  # Grid spacing in y
kappa = 0.01  # Thermal conductivity coefficient

# Initialize temperature field
u = np.zeros((nx, ny))

# Function to visualize the temperature field
def visualize_temperature(u, title="Temperature Field"):
    plt.imshow(u, cmap='hot', extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')
    plt.colorbar(label='Temperature')
    plt.title(title)
    plt.show()

# Main heat equation solver function
def heat_solver():
    # Solve the steady heat equation
    for _ in range(1000):  # Adjust the number of iterations as needed
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])

    # Visualize the temperature field
    #visualize_temperature(u)
    return u

# Run the heat solver
heat_solver()


#This is a simplified solver for a basic representation