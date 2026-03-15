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
size = 100
L = 1

global_solution = None


# Animation function
def animate(i):
    t, u = zip(*global_solution)
    line.set_ydata(u[i])
    return (line,)


class ExplicitHeatEquationProblem1D:
    def __init__(self, size, L):
        self.size = size
        self.h = L / (size - 1)

    def get_degrees_of_freedom(self):
        return self.size

    def set_parameters(self, size, L):
        self.size = size
        self.h = L / (size - 1)

    def set_initial_condition(self, u):
        for i in range(self.size):
            x = i * self.h
            u[i] = 1.0 if 0.4 < x < 0.6 else 0.0

    def function_f(self, t, u):
        fu = np.zeros(self.size)
        u[0] = 0.0
        u[self.size - 1] = 0.0
        fu[0] = 0.0
        fu[self.size - 1] = 0.0

        h_sqr = self.h * self.h
        for i in range(1, self.size - 1):
            fu[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / h_sqr

        return fu

    def plot_solution(self, steps, size, solutions, save_fig=False):
        x = np.linspace(0, 1, size)
        t, solution = zip(*solutions)

        for u in solution:
            plt.plot(x, u, label=f"Time {t}")

        plt.xlabel("Position")
        plt.ylabel("Temperature")
        plt.title("Heat Equation 1D Solution")
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(f"heat-equation.png")
        plt.show()


if __name__ == "__main__":

    problem = ExplicitHeatEquationProblem1D(size, L)

    u = np.zeros(size)
    problem.set_initial_condition(u)
    M = np.max(u)

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
    fig, ax = plt.subplots()
    (line,) = ax.plot(np.linspace(0, L, size), u[0])
    ax.set_ylim(0, 1.1 * M)
    ax.set_xlabel("Position along the rod")
    ax.set_ylabel("Temperature")
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
    plt.show()
