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
nu = 0.2  # Viscosity coefficient

# Initialize flow variables
u = np.zeros((nx, ny))  # x-component of velocity
v = np.zeros((nx, ny))  # y-component of velocity
p = np.zeros((nx, ny))  # Pressure

# Time parameters
dt = 0.001  # Time step
num_time_steps = 100

def initialize_flow():
    # Initialize flow variables with a clockwise-rotating vortex and an incoming flow
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    
    # Create a clockwise-rotating vortex
    vortex_radius = 0.15 * min(Lx, Ly)  # Adjust the vortex radius
    vortex_center = np.array([Lx / 2, Ly / 2])
    u_vortex = 2 * np.pi * (y - vortex_center[1]) * np.exp(-((x - vortex_center[0])**2 + (y - vortex_center[1])**2) / vortex_radius**2)
    v_vortex = -2 * np.pi * (x - vortex_center[0]) * np.exp(-((x - vortex_center[0])**2 + (y - vortex_center[1])**2) / vortex_radius**2)

    # Flow coming from the left, reduced size and increased strength
    u[:, :2*ny//3] = 2.5

    # Combine the vortex and incoming flow
    u += u_vortex
    v += v_vortex

    # Introduce interesting patterns in the temperature field
    a = 2 * np.pi / Lx
    b = 2 * np.pi / Ly
    temperature = np.sin(a * x) * np.cos(b * y) + 0.5 * np.sin(2 * a * x) * np.sin(2 * b * y)

    initial_temperature = 100.0
    temperature *= initial_temperature

    return u, v, temperature

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

    visualize_vectors(u, v, title="Navier–Stokes Flow Field - Initial Condition")

def visualize_vectors(u, v, title="Flow Field"):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Subsample the u and v arrays to reduce arrow density
    subsample_factor = 10
    u_subsampled = u[::subsample_factor, ::subsample_factor]
    v_subsampled = v[::subsample_factor, ::subsample_factor]

    quiver = ax.quiver(u_subsampled, v_subsampled, scale=30, width=0.005, color='red', alpha=0.7, headaxislength=3)
    ax.set_title(title)

    def update_quiver(num, quiver, u, v):
        quiver.set_UVC(u[::subsample_factor, ::subsample_factor], v[::subsample_factor, ::subsample_factor])
        return quiver,

    ani = FuncAnimation(fig, update_quiver, frames=num_time_steps, fargs=(quiver, u, v), interval=5000, blit=False)
    plt.show()

# Main Navier–Stokes solver function
def navier_stokes_solver():

    u, v, p = initialize_flow()

    visualize_flow(u, v, title="Navier–Stokes Flow Field - Initial Condition")

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

    visualize_flow(u, v, title="Navier–Stokes Flow Field - Result")

    return u, v, p