from NavierStokes import navier_stokes_solver, visualize_flow
from SteadyHeatEquationFFT import heat_solver, visualize_temperature

def main():
    # Run Navier–Stokes solver
    u, v, p = navier_stokes_solver()

    # Visualize the flow field
    visualize_flow(u, v, title="Navier–Stokes Flow Field")

    # Run heat solver
    temperature = heat_solver(u)

    # Visualize the temperature field
    visualize_temperature(temperature, title="Heat Map")


if __name__ == "__main__":
    main()