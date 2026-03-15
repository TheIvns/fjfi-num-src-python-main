# poisson2d.py
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from Linear_systems.stationary_solver import StationarySolver


def plot_solution_3d(solution, N, model):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    Z = solution.reshape((N + 1, N + 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Solution")
    plt.savefig(model + "_3d" + ".png", format="png", bbox_inches="tight")
    plt.show()


def plot_solution_2d(solution, N, model):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y)
    Z = solution.reshape((N + 1, N + 1))

    plt.figure()
    plt.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar(label="Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(model + "_2d" + ".png", format="png", bbox_inches="tight")
    plt.show()


def generate_Ab(n, rhs_name, boundary_name="const"):
    A = np.zeros((n * n, n * n))
    b = np.zeros(n * n)

    # todo - set the matrix A and boundary conditions

    return A, b


if __name__ == "__main__":
    N = 50
    h = 1.0 / (N - 1)
    h_sqr = h * h

    # Set-up the linear system
    dofs = (N + 1) * (N + 1)
    u = np.zeros(dofs)

    A, b = generate_Ab(N + 1, "linear", "const")

    start = time.time()
    method = "gauss-seidel"  # "sor", "jacobi", "richardson", "gauss-seidel"

    # Solve the linear system using StationarySolver
    solver = StationarySolver(A, b)
    solver.set_max_iterations(1000)
    solver.set_convergence_residue(1.0e-4)

    relaxation = 1.87  # todo - find optimal relaxation parameter

    solver.set_relaxation(relaxation)
    solution = solver.solve(method=method, verbose=True)
    stop = time.time()
    print(f"Solve time: {stop - start:0.2f} seconds")

    # Write the solution to the output file
    with open("poisson-2d.txt", "w") as f:
        for j in range(N + 1):
            for i in range(N + 1):
                index = j * (N + 1) + i
                f.write(f"{i * h} {j * h} {solution[index]}\n")

    plot_solution_2d(solution, N, "cc")  # c, l, q, s
    plot_solution_3d(solution, N, "cc")
