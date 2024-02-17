# ThomasTridiagonalSystem.py

import numpy as np

def thomas_algorithm(a, b, c, f):
    """
    Solve a tridiagonal system of linear equations using the Thomas algorithm.

    Parameters:
        a (numpy.ndarray): Lower diagonal elements.
        b (numpy.ndarray): Main diagonal elements.
        c (numpy.ndarray): Upper diagonal elements.
        f (numpy.ndarray): Right-hand side vector.

    Returns:
        numpy.ndarray: Solution vector.
    """
    n = len(f)
    
    # Initialize coefficients
    beta = np.zeros(n)
    gamma = np.zeros(n)
    
    # Compute coefficients using recurrence relation
    beta[0] = b[0]
    gamma[0] = f[0] / beta[0]
    
    for k in range(1, n):
        beta[k] = b[k] - c[k-1] / beta[k-1] * a[k-1]
        gamma[k] = (f[k] - a[k] * gamma[k-1]) / beta[k]
    
    # Backward substitution
    X = np.zeros(n)
    X[-1] = gamma[-1]
    
    for k in range(n-2, -1, -1):
        X[k] = (gamma[k] - c[k] * X[k+1]) / beta[k]
    
    return X

# Example usage
if __name__ == "__main__":
    # Example coefficients (replace these with your actual values)
    a = np.array([1.0, 2.0, 3.0, 4.0])  # Lower diagonal
    b = np.array([2.0, 3.0, 5.0, 7.0])  # Main diagonal
    c = np.array([1.0, 2.0, 3.0, 4.0])  # Upper diagonal
    f = np.array([10.0, 20.0, 30.0, 40.0])  # Right-hand side

    result = thomas_algorithm(a, b, c, f)
    print(result)
