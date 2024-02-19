import numpy as np
import matplotlib.pyplot as plt

# Global parameters
Lx = 1.0
Ly = 2.0
nx = 21
ny = 51
cfl = 1.0

# Define grid and step sizes
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize grid and solution
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
u0 = np.zeros((nx, ny))

# Function to compute Laplacian (∆u) using vectorized programming
def calc_lap(u):
    global dx, dy
    im = np.roll(u, 1, axis=0)
    ip = np.roll(u, -1, axis=0)
    jm = np.roll(u, 1, axis=1)
    jp = np.roll(u, -1, axis=1)

    laplacian = (ip - 2*u + im) / (dx**2) + (jp - 2*u + jm) / (dy**2)
    return laplacian

# Function to compute the right-hand side function f(x, y)
def compute_rhs(X, Y):
    a = 2 * np.pi / Lx
    b = 2 * np.pi / Ly
    return (a**2 + b**2) * np.sin(a * X) * np.cos(b * Y)

# Function to solve the unsteady heat equation using the explicit scheme
def solve_heat_equation(u0, cfl, max_iterations=10000, convergence_threshold=1e-6):
    u = u0.copy()
    dt = cfl * min(dx**2, dy**2) / 2  # Time step based on CFL condition

    iteration = 0
    while iteration < max_iterations:
        rhs = compute_rhs(X, Y)
        laplacian = calc_lap(u)
        u_new = u + dt * (rhs + laplacian)

        # Check convergence
        epsilon = np.linalg.norm(u_new - u) / np.sqrt(nx * ny)
        if epsilon < convergence_threshold:
            break

        u = u_new
        iteration += 1

    return u

# Plot the numerical and exact solutions
def plot_solutions(u, exact_solution):
    plt.contourf(X, Y, u.T, levels=20, cmap='viridis', extend='both')
    plt.colorbar(label='Numerical Solution')
    plt.contour(X, Y, exact_solution.T, levels=20, colors='w', linestyles='dashed')
    plt.title('Numerical vs Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Main program
if __name__ == "__main__":
    # Solve the unsteady heat equation
    numerical_solution = solve_heat_equation(u0, cfl)

    # Exact solution for comparison
    exact_solution = np.sin(2 * np.pi / Lx * X) * np.cos(2 * np.pi / Ly * Y)

    # Plot the solutions
    plot_solutions(numerical_solution, exact_solution)
