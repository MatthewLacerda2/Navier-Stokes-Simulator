import numpy as np
import matplotlib.pyplot as plt
from ThomasTridiagonalPeriodicSystem import NSE_trid_per_c2D as nse_trid

Lx = 1
Ly = 1
nx = 50
ny = 50

def visualize_steady_heat(x,y,u):
    plt.contourf(x, y, u, cmap='viridis')
    plt.title("Steady Heat Equation - FFT along x, FD along y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()

def visualize_temperature(u, title="Temperature Field"):
    plt.imshow(u, cmap='hot', extent=[0, Lx, 0, Ly], origin='lower', aspect='auto')
    plt.colorbar(label='Temperature')
    plt.title(title)
    plt.show()

# Function to solve the steady heat equation using FFT along x and finite differences along y
def heat_solver(f):

    # Constants for the right-hand side function (modify based on Exercise 12.2)
    a = 2 * np.pi / Lx
    b = 2 * np.pi / Ly

    # Optimize the tridiagonal system solver
    aa, ab, ac, fi = np.random.rand(ny, nx), np.random.rand(ny, nx), np.random.rand(ny, nx), np.random.rand(ny, nx)
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

    visualize_steady_heat(x,y,u)

    return u