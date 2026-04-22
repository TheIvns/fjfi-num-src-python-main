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
    Z = np.array(solution).reshape((N + 1, N + 1))

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
    Z = np.array(solution).reshape((N + 1, N + 1))

    plt.figure()
    plt.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar(label="Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(model + "_2d" + ".png", format="png", bbox_inches="tight")
    plt.show()


def generate_Ab(n, rhs_name, boundary_name="const"):
    N_total = n * n
    A = np.zeros((N_total, N_total))
    b = np.zeros(N_total)
    h = 1.0 / (n - 1)
    h_sqr = h * h

    for i in range(N_total):
        # Souřadnice v mřížce pro kontrolu okrajů
        row = i // n
        col = i % n
        

        if row == 0 or row == n-1 or col == 0 or col == n-1:
            A[i, i] = 1.0
            b[i] = 10.0  #boundary
    
        else:
            A[i, i] = -4.0
            A[i, i-1] = 1.0  
            A[i, i+1] = 1.0 
            A[i, i-n] = 1.0  
            A[i, i+n] = 1.0  
            
            b[i] = 10.0 * h_sqr 
            
    return A, b


if __name__ == "__main__":
    N = 50
    h = 1.0 /  (N - 1)
    h_sqr = h * h

    # Set-up the linear system
    dofs = (N + 1) * (N + 1)
    u = np.zeros(dofs)

    A, b = generate_Ab(N + 1, "linear", "const")

    start = time.time()
    method = "sor"  # "sor", "jacobi", "richardson", "gauss-seidel"

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
