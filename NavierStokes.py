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

# Modify the external force initialization
external_force_magnitude = 0.2  # Adjust the force magnitude as needed
force_width_fraction = 1/10.0  # Fraction of the screen width for the force region

# Calculate the width and center of the force region
force_width = int(nx * force_width_fraction)
force_center_x = int(nx / 2)

# Initialize flow variables
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))

nu = 0.2  # Viscosity coefficient
dt = 0.001  # Time step
num_time_steps = 10000

def visualize_flow(u, v, title="Flow Field"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate the magnitude of the velocity field
    speed = np.sqrt(u**2 + v**2)

    # Use a color map to represent the magnitude
    im = ax1.imshow(speed, cmap='viridis', extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax1, label='Speed')
    ax1.set_title(title)

    # Subsample the u and v arrays to reduce arrow density
    subsample_factor = 10
    u_subsampled = u[::subsample_factor, ::subsample_factor]
    v_subsampled = v[::subsample_factor, ::subsample_factor]

    quiver = ax2.quiver(u_subsampled, v_subsampled, scale=30, width=0.005, color='red', alpha=0.7, headaxislength=3)
    ax2.set_title("Vector Field")

    def update_quiver(num, quiver, u, v):
        quiver.set_UVC(u[::subsample_factor, ::subsample_factor], v[::subsample_factor, ::subsample_factor])
        return quiver,

    ani = FuncAnimation(fig, update_quiver, frames=num_time_steps, fargs=(quiver, u, v), interval=5000, blit=False)
    plt.show()

def visualize_external_forces(external_force, title="External Forces"):
    plt.figure(figsize=(8, 8))
    plt.quiver(external_force[:, :, 0], external_force[:, :, 1], scale=20, scale_units='xy', color='red')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Main Navier–Stokes solver function
def navier_stokes_solver(u, v, p):

    # Initialize external force
    external_force = np.zeros((nx, ny, 2))
    external_force[force_center_x:force_center_x + force_width, :, 0] = external_force_magnitude

    visualize_flow(u, v, title="Navier–Stokes Flow Field - Initial Condition")
    visualize_external_forces(external_force, title="External Forces Visualization")

    for t in range(num_time_steps):
        # Solve Navier–Stokes equations with external force (simplified)
        u[1:-1, 1:-1] += dt * (nu * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
                             + nu * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2)
        v[1:-1, 1:-1] += dt * (nu * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
                             + nu * (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2)

        # Pressure correction step
        p[1:-1, 1:-1] = 0.5 * (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2]) - \
                       dt / (2 * dx) * (u[2:, 1:-1] - u[:-2, 1:-1]) - \
                       dt / (2 * dy) * (v[1:-1, 2:] - v[1:-1, :-2])
        
        # Apply external force
        u[force_center_x:force_center_x + force_width, :] += dt * external_force[force_center_x:force_center_x + force_width, :, 0]
        v[force_center_x:force_center_x + force_width, :] += dt * external_force[force_center_x:force_center_x + force_width, :, 1]

    visualize_flow(u, v, title="Navier–Stokes Flow Field - Result")

    return u, v, p