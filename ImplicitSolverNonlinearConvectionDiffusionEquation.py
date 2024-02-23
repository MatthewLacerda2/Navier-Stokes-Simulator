import numpy as np
import matplotlib.pyplot as plt

def solve_nonlinear_convection_diffusion(Lx=1, Ly=2, nx=21, ny=51, cfl=100, t_end=1e6, epsilon=1e-6):
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

    # Define source term function for nonlinear convection-diffusion
    def fsource_nonlinear(x, y, u):
        a = 2 * np.pi / Lx
        b = 2 * np.pi / Ly
        return (a**2 + b**2) * np.sin(a * x) * np.cos(b * y) + u**2

    # ADI initialization
    ami_x = -dt / (2 * dx**2)
    api_x = -dt / (2 * dx**2)
    alph_x = 1 + dt / dx**2

    ami_y = -dt / (2 * dy**2)
    api_y = -dt / (2 * dy**2)
    alph_y = 1 + dt / dy**2

    while t < t_end:
        # ADI step in x-direction
        for j in range(1, ny-1):
            fi = u[:, j] + dt * fsource_nonlinear(x, y[j], u[:, j]) + ami_x * (u[:-2, j] - 2*u[1:-1, j] + u[2:, j])
            u[:, j] = np.linalg.solve(np.diag(api_x * np.ones(nx-1), -1) + np.diag(alph_x * np.ones(nx), 0) +
                                       np.diag(ami_x * np.ones(nx-1), 1), fi)

        # ADI step in y-direction
        for i in range(1, nx-1):
            fi = u[i, :] + dt * fsource_nonlinear(x[i], y, u[i, :]) + ami_y * (u[i, :-2] - 2*u[i, 1:-1] + u[i, 2:])
            u[i, :] = np.linalg.solve(np.diag(api_y * np.ones(ny-1), -1) + np.diag(alph_y * np.ones(ny), 0) +
                                       np.diag(ami_y * np.ones(ny-1), 1), fi)

        # Check convergence
        epsilon_t = np.linalg.norm(u - calc_lap(u))
        if epsilon_t < epsilon:
            break

        # Update time
        t += dt

    # Return numerical solution
    X, Y = np.meshgrid(x, y)
    numerical_solution_nonlinear = u.T

    return numerical_solution_nonlinear

if __name__ == "__main__":
    numerical_solution_nonlinear = solve_nonlinear_convection_diffusion()

    # Plot numerical solution for nonlinear convection-diffusion
    X, Y = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 2, 51))
    plt.contourf(X, Y, numerical_solution_nonlinear, levels=50, cmap='viridis')
    plt.title('Numerical Solution for Nonlinear Convection-Diffusion')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
