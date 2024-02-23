import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define global variables
Lx = 10  # Domain size in x
Ly = 10  # Domain size in y
nx = 250  # Number of grid points in x
ny = 250  # Number of grid points in y
dx = Lx / (nx - 1)  # Grid spacing in x
dy = Ly / (ny - 1)  # Grid spacing in y
nu = 0.25  # Viscosity coefficient

# Initialize flow variables
u = np.zeros((nx, ny))  # x-component of velocity
v = np.zeros((nx, ny))  # y-component of velocity
p = np.zeros((nx, ny))  # Pressure

# Time parameters
dt = 0.001  # Time step
num_time_steps = 1000

def initialize_flow():
    # Initialize flow variables with two flows colliding
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))

    # Control the amplitude of randomness based on the time_step
    randomness_factor = 0.1 + 0.1 * np.exp(-num_time_steps / 500)

    # Flow going from left to right with increasing randomness
    u[:, :ny//2] = 1.0 + randomness_factor

    # Flow going from top to bottom with increasing randomness
    v[:nx//2, :] = 1.0 + randomness_factor

    initial_pressure = 1000
    p = np.full((nx, ny), initial_pressure)

    return u, v, p

# Function to visualize the flow field
def visualize_flow(u, v, title="Flow Field"):
    fig, ax = plt.subplots()

    # Calculate the magnitude of the velocity field
    speed = np.sqrt(u**2 + v**2)

    # Use a color map to represent the magnitude
    im = ax.imshow(speed, cmap='viridis', extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')
    plt.colorbar(im, label='Speed')

    ax.set_title(title)
    plt.show()

# Main Navier–Stokes solver function
def navier_stokes_solver():

    u, v, p = initialize_flow()

    for t in range(num_time_steps):
        # Solve Navier–Stokes equations (simplified)
        u[1:-1, 1:-1] += dt * (nu * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
                             + nu * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2)

        v[1:-1, 1:-1] += dt * (nu * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
                             + nu * (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2)

        # Pressure correction step
        p[1:-1, 1:-1] = 0.5 * (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2]) - \
                       dt / (2 * dx) * (u[2:, 1:-1] - u[:-2, 1:-1]) - \
                       dt / (2 * dy) * (v[1:-1, 2:] - v[1:-1, :-2])

    # Return u, v, and p
    return u, v, p