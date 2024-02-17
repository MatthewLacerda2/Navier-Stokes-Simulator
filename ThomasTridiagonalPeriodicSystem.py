# ThomasTridiagonalPeriodicSystem.py

import numpy as np
from ThomasTridiagonalSystem import thomas_algorithm  # Import the previous Thomas algorithm

def thomas_periodic_algorithm(a, b, c, f):
    """
    Solve a periodic tridiagonal system of linear equations using the Thomas algorithm.

    Parameters:
        a (numpy.ndarray): Lower diagonal elements.
        b (numpy.ndarray): Main diagonal elements.
        c (numpy.ndarray): Upper diagonal elements.
        f (numpy.ndarray): Right-hand side vector.

    Returns:
        numpy.ndarray: Final solution vector.
    """
    n = len(f)

    # Step 1: Solve two tridiagonal systems using the Thomas algorithm
    X1 = thomas_algorithm(a, b, c, f)
    X2 = thomas_algorithm(a, b, c, np.concatenate(([a[0]], np.zeros(n-1))))

    # Step 2: Compute X* from equation (12.71)
    X_star = X1[0] + X1[-1] * (X2[0] / X2[-1]) / (1 + X2[0] / X2[-1])

    # Step 3: Compute the final solution using equation (12.69)
    final_solution = X1 + X_star * np.concatenate(([1], np.zeros(n-1)))

    return final_solution

# Example usage
if __name__ == "__main__":
    # Example coefficients (replace these with your actual values)
    a = np.array([1.0, 2.0, 3.0, 4.0])  # Lower diagonal
    b = np.array([2.0, 3.0, 5.0, 7.0])  # Main diagonal
    c = np.array([1.0, 2.0, 3.0, 4.0])  # Upper diagonal
    f = np.array([10.0, 20.0, 30.0, 40.0])  # Right-hand side

    result = thomas_periodic_algorithm(a, b, c, f)
    print(result)
