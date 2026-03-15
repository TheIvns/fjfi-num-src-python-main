import numpy as np

def power_method(matrix: np.ndarray, max_iterations: int = 1000, tolerance: float = 1e-10):
    """
    Finds the largest eigenvalue and corresponding eigenvector using the Power Method.
    """
    n = matrix.shape[0]

    x = np.random.rand(n)  # Start with a random vector
    x = x / np.linalg.norm(x)  # Normalize the initial vector

    eigenvalue = 0

    for _ in range(max_iterations):
        x_new = np.dot(matrix, x) # Multiply matrix by current vector
        eigenvalue_new = np.dot(x_new, x) / np.dot(x, x) # Approximate the eigenvalue as the quotient
        x_new = x_new / np.linalg.norm(x_new)  # Normalize the new vector

        # Check for convergence
        if np.abs(eigenvalue_new - eigenvalue) < tolerance:
            x = x_new
            eigenvalue = eigenvalue_new
            break

        x = x_new
        eigenvalue = eigenvalue_new

    return eigenvalue, x

def read_matrix_from_file(file_name: str):
    try:
        matrix = np.loadtxt(file_name)
    except FileNotFoundError:
        print(f"Cannot open file {file_name}.")
        return None
    return matrix

if __name__ == "__main__":
    file_name = "matrix.txt"
    matrix = read_matrix_from_file(file_name)

    largest_eigenvalue, corresponding_eigenvector = power_method(matrix)
    print("Largest Eigenvalue:", largest_eigenvalue)
    print("Corresponding Eigenvector:", corresponding_eigenvector)
    # np.savetxt("largest_eigenvector.txt", corresponding_eigenvector)