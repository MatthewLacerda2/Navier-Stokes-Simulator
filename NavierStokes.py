import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define global variables
Lx = 1.0  # Domain size in x
Ly = 1.0  # Domain size in y
nx = 50   # Number of grid points in x
ny = 50   # Number of grid points in y
dx = Lx / (nx - 1)  # Grid spacing in x
dy = Ly / (ny - 1)  # Grid spacing in y
nu = 0.1  # Viscosity coefficient

# Initialize flow variables
u = np.zeros((nx, ny))  # x-component of velocity
v = np.zeros((nx, ny))  # y-component of velocity
p = np.zeros((nx, ny))  # Pressure

# Time parameters
dt = 0.001  # Time step
num_time_steps = 100

# Function to visualize the flow field
def visualize_flow(u, v, title="Flow Field"):
    fig, ax = plt.subplots()
    quiver = ax.quiver(u, v, scale=20)
    ax.set_title(title)
    
    def update_quiver(num, quiver, u, v):
        quiver.set_UVC(u, v)
        return quiver,

    ani = FuncAnimation(fig, update_quiver, frames=num_time_steps, fargs=(quiver, u, v), interval=50, blit=False)
    plt.show()

# Main Navier–Stokes solver function
def navier_stokes_solver():
    for t in range(num_time_steps):
        # Solve Navier–Stokes equations (simplified)
        u[1:-1, 1:-1] += dt * (nu * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
                             + nu * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2)

        v[1:-1, 1:-1] += dt * (nu * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
                             + nu * (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2)

        p[1:-1, 1:-1] = 0  # Placeholder for pressure computation

    # Return u and v
    return u, v

#This is a simplified solver for a basic representation