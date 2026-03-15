import numpy as np


def gram_schmidt_QR(A):
    """
    Computes the QR decomposition of matrix A using the Gram-Schmidt process.

    Args:
        A: A square matrix.

    Returns:
        Q: Orthonormal matrix (Q such that A = Q * R).
        R: Upper triangular matrix (R such that A = Q * R).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Start with the j-th column of A
        v = A[:, j]

        # Subtract the projection onto the previously computed Q columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]

        # Normalize v to get the orthonormal vector
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def householder_transformation_QR(A):
    """
    Computes the QR decomposition of a matrix A using Householder transformations.

    Args:
        A: The input matrix (m x n).

    Returns:
        Q: Orthonormal matrix (m x m).
        R: Upper triangular matrix (m x n).
    """
    m, n = A.shape
    Q = np.eye(m)  # Initialize Q as the identity matrix
    R = np.copy(A)  # Start with R as the copy of A

    for k in range(n):
        # Step 1: Compute the Householder vector v
        x = R[k:m, k]  # The current column of R starting from row k
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)  # The first component of e1 is the norm of x

        v = x + np.sign(x[0]) * e1  # The vector v for the Householder transformation
        v = v / np.linalg.norm(v)  # Normalize v

        # Step 2: Apply the Householder transformation H
        H = np.eye(m - k) - 2 * np.outer(v, v)  # Householder matrix for column k

        # Apply H to the matrix R from the left (only affecting rows from k onwards)
        R[k:m, k:n] = H @ R[k:m, k:n]

        # Update Q by accumulating the Householder transformations
        Q_k = np.eye(m)
        Q_k[k:m, k:m] = H
        Q = Q @ Q_k.T  # Accumulate the transformations into Q

    return Q, R

def qr_algorithm(matrix: np.ndarray, max_iterations: int = 1000, tolerance: float = 1e-10):
    """
    Uses the QR algorithm to approximate eigenvalues of the matrix.
    """
    A = np.copy(matrix)

    for _ in range(max_iterations):
        # Perform QR decomposition
        Q, R = np.linalg.qr(A)  # pre implemented using numpy

        # Custom implementations
        # Q, R = gram_schmidt_QR(A)
        # Q, R = householder_transformation_QR(A)

        # Update A by R @ Q (A = R * Q)
        A = R @ Q

        # Check for convergence by looking at the off-diagonal elements
        off_diagonal_sum = np.sum(np.abs(A - np.diag(np.diag(A))))

        if off_diagonal_sum < tolerance:
            break

    # Eigenvalues are approximated on the diagonal of A
    eigenvalues = np.diag(A)
    return eigenvalues

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
    if matrix is not None:
        n, m = matrix.shape
        if n != m:
            print("Only square matrices are allowed for eigenvalue computation.")
        else:
            eigenvalues = qr_algorithm(matrix)
            print("Approximated Eigenvalues:", eigenvalues)
            # np.savetxt("eigenvalues.txt", eigenvalues)
