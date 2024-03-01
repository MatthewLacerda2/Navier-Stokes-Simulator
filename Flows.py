#Simply some flows to use
import numpy as np

# Define global variables
Lx = 10  # Domain size in x
Ly = 10  # Domain size in y
nx = 250  # Number of grid points in x
ny = 250  # Number of grid points in y

def initialize_flow_1():
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

def initialize_flow_2():
    # Initialize flow variables with two flows colliding
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))

    # Control the amplitude of randomness based on the time_step
    randomness_factor = 0.1 + 0.1

    # Flow going from left to right
    u[:, :ny//2] = 2.5

    # Flow going from bottom to top (1.5 times the speed of left-to-right flow)
    v[:nx//2, :] = 1.5

    a = 2 * np.pi / Lx
    b = 2 * np.pi / Ly
    temperature = np.sin(a * x) * np.cos(b * y) + 0.5 * np.sin(2 * a * x) * np.sin(2 * b * y)

    initial_temperature = 100.0
    temperature *= initial_temperature

    return u, v, temperature

def initialize_flow_3():
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))

    # Create a combination of vortices and flows
    vortex1_radius = 0.2 * min(Lx, Ly)
    vortex2_radius = 0.15 * min(Lx, Ly)
    vortex1_center = np.array([Lx / 3, Ly / 3])
    vortex2_center = np.array([2 * Lx / 3, 2 * Ly / 3])

    # Vortex 1
    u_vortex1 = 2 * np.pi * (y - vortex1_center[1]) * np.exp(-((x - vortex1_center[0])**2 + (y - vortex1_center[1])**2) / vortex1_radius**2)
    v_vortex1 = -2 * np.pi * (x - vortex1_center[0]) * np.exp(-((x - vortex1_center[0])**2 + (y - vortex1_center[1])**2) / vortex1_radius**2)

    # Vortex 2
    u_vortex2 = 2 * np.pi * (y - vortex2_center[1]) * np.exp(-((x - vortex2_center[0])**2 + (y - vortex2_center[1])**2) / vortex2_radius**2)
    v_vortex2 = -2 * np.pi * (x - vortex2_center[0]) * np.exp(-((x - vortex2_center[0])**2 + (y - vortex2_center[1])**2) / vortex2_radius**2)

    # Flow coming from the left
    u[:, :ny//2] = 1.5

    # Combine vortices and flows
    u += u_vortex1 + u_vortex2
    v += v_vortex1 + v_vortex2
    vortex_radius = 0.2 * min(Lx, Ly)

    temperature = 50.0 * np.exp(-((x - Lx/2)**2 + (y - Ly/2)**2) / (2 * (vortex_radius/2)**2))

    return u, v, temperature

def initialize_flow_4():
    # Initialize flow variables with a vortex dipole and additional flow
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))

    # Create a vortex dipole
    vortex_radius = 0.2 * min(Lx, Ly)  # Adjust the vortex radius
    vortex_center1 = np.array([Lx / 3, Ly / 2])
    vortex_center2 = np.array([2 * Lx / 3, Ly / 2])

    u_vortex1 = 2 * np.pi * (y - vortex_center1[1]) * np.exp(-((x - vortex_center1[0])**2 + (y - vortex_center1[1])**2) / vortex_radius**2)
    v_vortex1 = -2 * np.pi * (x - vortex_center1[0]) * np.exp(-((x - vortex_center1[0])**2 + (y - vortex_center1[1])**2) / vortex_radius**2)

    u_vortex2 = 2 * np.pi * (y - vortex_center2[1]) * np.exp(-((x - vortex_center2[0])**2 + (y - vortex_center2[1])**2) / vortex_radius**2)
    v_vortex2 = -2 * np.pi * (x - vortex_center2[0]) * np.exp(-((x - vortex_center2[0])**2 + (y - vortex_center2[1])**2) / vortex_radius**2)

    # Additional flow passing through
    u[:, ny//3:2*ny//3] = 1.5

    # Combine the vortex dipole and additional flow
    u += u_vortex1 + u_vortex2
    v += v_vortex1 + v_vortex2

    a = 2 * np.pi / Lx
    b = 2 * np.pi / Ly
    temperature = np.sin(a * x) * np.cos(b * y) + 0.5 * np.sin(2 * a * x) * np.sin(2 * b * y)

    initial_temperature = 100.0
    temperature *= initial_temperature

    return u, v, temperature