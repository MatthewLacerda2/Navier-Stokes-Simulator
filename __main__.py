from NavierStokes import navier_stokes_solver
from SteadyHeatEquationFFT import heat_solver, visualize_temperature
import Flows

def get_flow_choice():
    print("Which flow would you like to simulate:")
    print("1 - Flow on Vortex collision")
    print("2 - Perpendicular Flows")
    print("3 - Vortex on Vortex collision")
    print("4 - Flow on Vortex Dipole collision")

    choice = input("Enter the number (1, 2, 3 or 4): ")
    return choice

def main():
    # Get user input for selecting the flow type
    flow_choice = get_flow_choice()

    # Run Navier–Stokes solver based on user choice
    if flow_choice == '1':
        u, v, temperature = Flows.initialize_flow_1()
    elif flow_choice == '2':
        u, v, temperature = Flows.initialize_flow_2()
    elif flow_choice == '3':
        u, v, temperature = Flows.initialize_flow_3()
    elif flow_choice == '4':
        u, v, temperature = Flows.initialize_flow_4()
    else:
        print("Invalid choice. Aborting program.")
        return

    navier_stokes_solver(u, v, temperature)

    # Run heat solver
    temperature = heat_solver(u, v)

    # Visualize the temperature field
    visualize_temperature(temperature, title="Heat Map")

if __name__ == "__main__":
    main()
