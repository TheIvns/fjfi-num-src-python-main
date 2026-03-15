import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys

sys.path.append("..")
from RungeKutta import RK_second_order
from Euler import Euler
from Merson import Merson
from ODE import solve_loop


class HyperbolicProblem:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def get_degrees_of_freedom(self):
        return 2

    def function_f(self, time, u, fu=None):  # harmonic oscillator
        u1, u2 = u
        return np.array([u2, -u1 - self.epsilon * u1**2 * u2])

    def plot_solution(self, solution):
        t_values, data = zip(*solution)
        u = np.array(data)
        u1 = u[:, 0]
        u2 = u[:, 1]

        plt.plot(t_values, u1, label="position")
        plt.plot(t_values, u2, label="speed")
        plt.xlabel("Time")
        plt.ylabel("Solution")
        plt.title("Hyperbolic Problem Solution")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    problem = HyperbolicProblem()

    integrator = Euler()
    # integrator = RK_second_order()
    # integrator = Merson()

    problem.epsilon = 0.0
    initial_time = 0.0
    final_time = 100.0
    time_step = 0.1
    integration_time_step = 0.1
    initial_conditions = [0.0, 10.0]

    numerical_solution = solve_loop(
        initial_time,
        final_time,
        time_step,
        integration_time_step,
        problem,
        integrator,
        initial_conditions,
    )
    problem.plot_solution(numerical_solution)
