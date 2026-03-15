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
final_time = 0.75
size = 100
L = 2
integration_time_step = 0.1 * L / size
time_step = integration_time_step

global_solution = None

# scheme = "fdm"
# scheme = "lax-fridrichs"
scheme = "upwind"

# initial_condition = "discontinuous"
initial_condition = "smooth"


# Animation function
def animate(i):
    t, u = zip(*global_solution)
    line.set_ydata(u[i])
    return (line,)


class ExplicitBurgersEquationProblem1D:
    def __init__(self, size, L):
        self.size = size
        self.h = L / (size - 1)

    def get_degrees_of_freedom(self):
        return self.size

    def set_parameters(self, size, L):
        self.size = size
        self.h = L / (size - 1)

    def set_initial_condition(self, u):
        if initial_condition == "discontinuous":
            for i in range(self.size):
                x = i * self.h
                u[i] = 0 if x < 0.5 else 1.0
        elif initial_condition == "smooth":
            for i in range(self.size):
                x = i * self.h - 1
                u[i] = -math.tanh(5 * x)

    def function_f(self, t, u):
        fu = np.zeros(self.size)
        fu[0] = 0.0
        fu[self.size - 1] = 0.0

        tau = integration_time_step
        if scheme == "fdm":
            for i in range(1, self.size - 1):
                fu[i] = -u[i] * (u[i + 1] - u[i - 1]) / (2 * self.h)
        elif scheme == "lax-fridrichs":
            for i in range(1, self.size - 1):
                fu[i] = -u[i] * (u[i + 1] - u[i - 1]) / (2 * self.h) + (
                    u[i + 1] - 2 * u[i] + u[i - 1]
                ) / (2 * tau)
        elif scheme == "upwind":
            for i in range(1, self.size - 1):
                if u[i] > 0:
                    fu[i] = -u[i] * (u[i] - u[i - 1]) / self.h
                else:
                    fu[i] = -u[i] * (u[i + 1] - u[i]) / self.h
        return fu

    def plot_solution(self, steps, size, solutions, save_fig=False):
        x = np.linspace(0, 1, size)
        t, solution = zip(*solutions)

        for u in solution:
            plt.plot(x, u, label=f"Time {t}")

        plt.xlabel("Position")
        plt.ylabel("Mass")
        plt.title("Burgers Equation 1D Solution")
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(f"Burgers-equation.png")
        plt.show()


if __name__ == "__main__":

    problem = ExplicitBurgersEquationProblem1D(size, L)

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
    ax.set_ylim(-1.1 * M, 1.1 * M)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mass")
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
    plt.show()
