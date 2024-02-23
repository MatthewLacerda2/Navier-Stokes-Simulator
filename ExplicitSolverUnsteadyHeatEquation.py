import numpy as np
import matplotlib.pyplot as plt

def solve_unsteady_heat_equation(Lx=1, Ly=2, nx=21, ny=51, cfl=1, t_end=1e6, epsilon=1e-6):
    # Define grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize solution matrix
    u = np.zeros((nx, ny))

    # Initialize time variables
    dt = cfl * min(dx**2, dy**2) / 2  # Time step
    t = 0

    # Define Laplacian function
    def calc_lap(u):
        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1] = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
        laplacian[1:-1, 1:-1] += (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        return laplacian

    # Define source term function
    def fsource(x, y):
        a = 2 * np.pi / Lx
        b = 2 * np.pi / Ly
        return (a**2 + b**2) * np.sin(a * x) * np.cos(b * y)

    # Main time-stepping loop
    while t < t_end:
        # Update solution using explicit scheme
        laplacian = calc_lap(u)
        source_term = fsource(x[1:-1, None], y[None, 1:-1])  # calculate source term only once

        u[1:-1, 1:-1] += dt * (source_term + laplacian[1:-1, 1:-1])

        # Check for invalid values
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            print("Invalid values encountered. Aborting.")
            break

        # Check convergence
        epsilon_t = np.linalg.norm(u - laplacian)
        if epsilon_t < epsilon:
            break

        # Update time
        t += dt

    # Return numerical and exact solutions
    X, Y = np.meshgrid(x, y)
    numerical_solution = u.T
    exact_solution = np.sin(2 * np.pi / Lx * X) * np.cos(2 * np.pi / Ly * Y) * np.exp(-t)

    return numerical_solution, exact_solution

if __name__ == "__main__":
    numerical_solution, exact_solution = solve_unsteady_heat_equation()
    
    # Plot numerical and exact solutions
    X, Y = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 2, 51))
    plt.contourf(X, Y, numerical_solution, levels=50, cmap='viridis')
    plt.title('Numerical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    # Plot exact solution
    plt.contourf(X, Y, exact_solution, levels=50, cmap='viridis')
    plt.title('Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
