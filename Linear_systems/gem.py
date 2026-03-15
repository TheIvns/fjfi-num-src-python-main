import numpy as np

class GEM:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        assert A.shape[0] == A.shape[1] == b.shape[0], "Matrix A must be square and have the same size as b"
        self.A = A.astype(float)
        self.b = b.astype(float)

    def solve(self, x: np.ndarray, verbose: int = 0) -> bool:
        n = self.A.shape[0]
        assert self.b.shape[0] == x.shape[0], "Dimension mismatch between b and x"

        if verbose:
            self.print_matrix()

        for k in range(n):
            if verbose:
                print(f"Elimination: {k}/{n}")
            pivot = self.A[k, k]
            if pivot == 0.0:
                print(f"Zero pivot has appeared in {k}-th step. GEM has failed.")
                return False
            self.b[k] /= pivot
            self.A[k, k:] /= pivot
            self.A[k, k] = 1.0

            if verbose > 1:
                print("Dividing by the pivot ... ")
                self.print_matrix()

            for i in range(k + 1, n):
                self.b[i] -= self.A[i, k] * self.b[k]
                self.A[i, k:] -= self.A[i, k] * self.A[k, k:]

            if verbose > 1:
                print(f"Subtracting the {k}-th row from the rows below ... ")
                self.print_matrix()

        for k in range(n - 1, -1, -1):
            x[k] = self.b[k] - np.dot(self.A[k, k + 1:], x[k + 1:])
        return True

    def solve_with_pivoting(self, x: np.ndarray, verbose: int = 0) -> bool:
        n = self.A.shape[0]
        assert self.b.shape[0] == x.shape[0], "Dimension mismatch between b and x"

        if verbose:
            self.print_matrix()

        for k in range(n):
            if verbose:
                print(f"Step {k}/{n}.... \r")

            pivot_position = np.argmax(np.abs(self.A[k:, k])) + k
            if pivot_position != k:
                self.A[[k, pivot_position]] = self.A[[pivot_position, k]]
                self.b[k], self.b[pivot_position] = self.b[pivot_position], self.b[k]

            if verbose > 1:
                print("\nChoosing element at {}-th row as pivot...".format(pivot_position))
                print("Swapping {}-th and {}-th rows ... ".format(k, pivot_position))
                self.print_matrix()

            pivot = self.A[k, k]
            if pivot == 0.0:
                print(f"Zero pivot has appeared in {k}-th step. GEM has failed.")
                return False
            self.b[k] /= pivot
            self.A[k, k:] /= pivot
            self.A[k, k] = 1.0

            if verbose > 1:
                print("Dividing by the pivot ... ")
                self.print_matrix()

            for i in range(k + 1, n):
                self.b[i] -= self.A[i, k] * self.b[k]
                self.A[i, k:] -= self.A[i, k] * self.A[k, k:]

            if verbose > 1:
                print(f"Subtracting the {k}-th row from the rows below ... ")
                self.print_matrix()

        for k in range(n - 1, -1, -1):
            x[k] = self.b[k] - np.dot(self.A[k, k + 1:], x[k + 1:])
        return True

    def compute_lu_decomposition(self, verbose: int = 0) -> bool:
        n = self.A.shape[0]
        if verbose:
            self.print_matrix()

        for k in range(n):
            pivot = self.A[k, k]
            self.b[k] /= pivot
            self.A[k, k:] /= pivot

            if verbose > 1:
                print("Dividing by the pivot ... ")
                self.print_matrix()

            for i in range(k + 1, n):
                self.b[i] -= self.A[i, k] * self.b[k]
                self.A[i, k:] -= self.A[i, k] * self.A[k, k:]

            if verbose > 1:
                print(f"Subtracting the {k}-th row from the rows below ... ")
                self.print_matrix()
        return True

    def print_matrix(self, precision=6):
        n = self.A.shape[0]
        zero = "."  # Symbol to use for zero values

        # Loop over each row to print
        for row in range(n):
            row_values = []
            for value in self.A[row]:
                if value == 0:
                    row_values.append(f"{zero:^{precision + 4}}")  # Handle zero as string
                else:
                    row_values.append(f"{value:^{precision + 4}.{precision}f}")  # Handle numerical values

            # Join row values and format the corresponding b value
            row_str = " ".join(row_values)
            print(f"| {row_str} | {self.b[row]:^{precision + 4}.{precision}f} |")


if __name__ == "__main__":
    # Define the coefficient matrix A and the right-hand side vector b
    A = np.array([
        [2, -1, 1],
        [3, 3, 9],
        [3, 3, 5]
    ], dtype=float)

    b = np.array([1, 0, 4], dtype=float)

    # Initialize the GEM solver with A and b
    gem_solver = GEM(A, b)

    # Prepare a vector to store the solution
    x = np.zeros(b.shape)

    # Solve the system using Gaussian Elimination with partial pivoting
    if gem_solver.solve_with_pivoting(x, verbose=2):
        print("Solution found:")
        print(x)
    else:
        print("No solution exists or the system is singular.")