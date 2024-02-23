import numpy as np

def thomas_periodic_algorithm(a, b, c, f):
    # Function to solve a tridiagonal periodic system using Thomas algorithm
    n = len(f)
    gamma = np.zeros(n)
    beta = np.zeros(n)

    # Initialization
    beta[0] = b[0]
    gamma[0] = f[0] / beta[0]

    # Forward sweep
    for k in range(1, n):
        beta[k] = b[k] - c[k-1] * a[k-1] / beta[k-1]
        gamma[k] = (f[k] - a[k-1] * gamma[k-1]) / beta[k]

    # Backward sweep
    X = np.zeros(n)
    X[-1] = gamma[-1]

    for k in range(n-2, -1, -1):
        X[k] = gamma[k] - c[k] * X[k+1] / beta[k]

    return X

def NSE_trid_per_c2D(aa, ab, ac, fi):
    m, n = aa.shape
    X = np.zeros((m, n))  # Initialize the solution matrix

    # Use vectorized programming to apply the relations of the algorithm simultaneously to all m systems
    ab[:, 0] -= aa[:, 0]
    ab[:, -1] -= ac[:, -1]

    # Loop over each system j
    for j in range(m):
        # Solve the tridiagonal system for the current system using thomas_periodic_algorithm
        X[j] = thomas_periodic_algorithm(aa[j], ab[j], ac[j], fi[j])

    return X

# Example usage
if __name__ == "__main__":
    # Example coefficients for multiple systems (replace these with your actual values)
    aa = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])  # Lower diagonal for each system
    ab = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])  # Main diagonal for each system
    ac = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Upper diagonal for each system
    fi = np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]])  # Right-hand side for each system

    result = NSE_trid_per_c2D(aa, ab, ac, fi)
    print(result)
