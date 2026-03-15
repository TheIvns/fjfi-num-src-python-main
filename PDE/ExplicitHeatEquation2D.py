import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

import sys
from os.path import dirname

sys.path.append("..")

from Euler import Euler
from Merson import Merson
from RungeKutta import RK_second_order
from ODE.ODE import *

initial_time = 0.0
final_time = 0.01
time_step = 0.00001
integration_time_step = 0.000001
size = 50
L = 1

global_solution = None
global_M = None


# Animation function
def animate(i):
    t, u = zip(*global_solution)
    cax.set_array(u[i])
    ax2.clear()
    ax2.plot_surface(X, Y, u[i], cmap="hot")
    ax2.set_zlim(0, 1.1 * global_M)
    ax2.set_title("Temperature 3D Surface Plot")
    ax2.set_xlabel("x position")
    ax2.set_ylabel("y position")
    ax2.set_zlabel("Temperature")
    return cax, ax2


class ExplicitHeatEquationProblem2D:
    def __init__(self, size, L):
        self.size = size
        self.h = L / (size - 1)

    def get_degrees_of_freedom(self):
        return self.size * self.size

    def set_parameters(self):
        pass

    def set_initial_condition(self, u):
        for i in range(self.size):
            for j in range(self.size):
                x = i * self.h
                y = j * self.h
                u[i, j] = 1.0 if 0.4 < x < 0.6 and 0.4 < y < 0.6 else 0.0

    def function_f(self, t, u):
        fu = np.zeros((self.size, self.size))
        # TODO: Write the numerical scheme for the heat equation
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                fu[i, j] = (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) / self.h**2 + (
                    u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]
                ) / self.h**2

        return fu


if __name__ == "__main__":
    problem = ExplicitHeatEquationProblem2D(size, L)

    u = np.zeros((size, size))
    problem.set_initial_condition(u)
    global_M = np.max(u)

    integrator = Euler()
    global_solution = solve_loop(
        initial_time,
        final_time,
        time_step,
        integration_time_step,
        problem,
        integrator,
        u,
    )

    # Animate the solution
    t, u = zip(*global_solution)
    fig = plt.figure(figsize=(14, 6))

    # Heatmap subplot
    ax1 = fig.add_subplot(121)
    cax = ax1.imshow(u[0], cmap="hot", origin="lower", extent=[0, L, 0, L])
    ax1.set_title("Temperature Heatmap")
    ax1.set_xlabel("x position")
    ax1.set_ylabel("y position")
    fig.colorbar(cax, ax=ax1)

    # 3D surface plot subplot
    ax2 = fig.add_subplot(122, projection="3d")
    x = np.linspace(0, L, size)
    y = np.linspace(0, L, size)
    X, Y = np.meshgrid(x, y)
    surf = ax2.plot_surface(X, Y, u[0], cmap="hot")
    ax2.set_zlim(0, 1.1 * global_M)
    ax2.set_title("Temperature 3D Surface Plot")
    ax2.set_xlabel("x position")
    ax2.set_ylabel("y position")
    ax2.set_zlabel("Temperature")
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=False)
    plt.tight_layout()
    plt.show()
