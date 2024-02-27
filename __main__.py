from NavierStokes import navier_stokes_solver, visualize_flow
from SteadyHeatEquationFFT import heat_solver, visualize_temperature
from Flows import initialize_flow_1, initialize_flow_2, initialize_flow_3

def get_flow_choice():
    print("Which flow would you like to simulate:")
    print("1 - Flow on Vortex collision")
    print("2 - Perpendicular Flows")
    print("3 - Vortex on Vortex collision")

    choice = input("Enter the number (1, 2, or 3): ")
    return choice

def main():
    # Get user input for selecting the flow type
    flow_choice = get_flow_choice()

    # Run Navier–Stokes solver based on user choice
    if flow_choice == '1':
        u, v, temperature = initialize_flow_1()
    elif flow_choice == '2':
        u, v, temperature = initialize_flow_2()
    elif flow_choice == '3':
        u, v, temperature = initialize_flow_3()
    else:
        print("Invalid choice. Aborting program.")
        return

    navier_stokes_solver(u,v, temperature)

    # Run heat solver
    temperature = heat_solver(u, v)

    # Visualize the temperature field
    visualize_temperature(temperature, title="Heat Map")

if __name__ == "__main__":
    main()
