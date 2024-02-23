import numpy as np
import matplotlib.pyplot as plt
from ThomasTridiagonalPeriodicSystem import NSE_trid_per_c2D as nse_trid

Lx = 10
Ly = 10
nx = 250
ny = 250

def visualize_temperature(u, title="Temperature Field"):
    plt.imshow(u, cmap='hot', extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')
    #plt.imshow(u, cmap='viridis', extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')

    plt.colorbar(label='Temperature')
    plt.title(title)
    plt.show()

def initialize_temperature(nx, ny, Lx, Ly, u, v, a_factor=2, b_factor=2, temperature_factor=0.1):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    x, y = np.meshgrid(x, y)

    a = a_factor * 2 * np.pi / Lx
    b = b_factor * 2 * np.pi / Ly

    # Use the velocity components to influence the temperature initialization
    f = temperature_factor * (u**2 + v**2)

    # Add multiple sinusoidal terms for non-homogeneity
    f += 0.5 * np.sin(2 * a * x) * np.cos(b * y)
    f += 0.2 * np.sin(3 * a * x + b * y)
    f += 0.3 * np.sin(4 * a * x - 2 * b * y)

    return f

# Function to solve the steady heat equation using FFT along x and finite differences along y
def heat_solver(u, v):

    f = initialize_temperature(nx, ny, Lx, Ly, u, v)

    # Constants for the right-hand side function
    a = 2 * np.pi / Lx
    b = 2 * np.pi / Ly

    # Optimize the tridiagonal system solver
    aa, ab, ac, fi = np.random.rand(ny, nx), np.random.rand(ny, nx),np.random.rand(ny, nx), np.random.rand(ny, nx)
    nse_trid(aa, ab, ac, fi)
    
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Define the grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Initialize the right-hand side (modify this based on your exact solution)
    for i in range(nx):
        for j in range(ny):
            f[j, i] = (a**2 + b**2) * np.sin(a * x[i]) * np.cos(b * y[j])

    # Solve the tridiagonal system (Exercise 12.3 optimization)
    u = nse_trid(aa, ab, ac, f)

    return u