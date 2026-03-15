import math
import numpy as np
import matplotlib.pyplot as plt
import sys

#potřeba měnit parametry úlohy, řešení je nespojité, tak je potřeba vybrat spojitý interval

sys.path.append("..")
from ODE import solve_loop
from merson import Merson
from euler import Euler
from RungeKutta import RK_second_order


class Ricatti:
    def get_degrees_of_freedom(self):
        return 1

    def function_f(self, t, u_):
        u = u_[0]
        fu = t ** (-4.0) * math.exp(t) + u + 2.0 * math.exp(-t) * u**2
        return np.array(fu)

    def get_exact_solution(self, t, c=1):
        sqrt_2 = math.sqrt(2.0)
        return math.exp(t) * (
            1.0 / (sqrt_2 * t**2) * math.tan(sqrt_2 * (c - 1.0 / t)) - 1.0 / (2.0 * t)
        )

    def get_exact_solutions(self, initial_time, final_time, time_step, c=1.0):
        solutions = []
        t = initial_time
        while t < final_time:
            solutions.append((t, self.get_exact_solution(t, c)))
            t = min(t + time_step, final_time)
        return solutions

    def plot_solution(self, exact_solutions, numerical_solutions, text):
        time_values_exact, solution_values_exact = zip(*exact_solutions)
        time_values_numerical, solution_values_numerical = zip(*numerical_solutions)

        # Plot the data in a separate window
        plt.figure(figsize=(10, 6))
        plt.plot(
            time_values_exact, solution_values_exact, label=f"Exact Solution", color="r"
        )
        plt.plot(
            time_values_numerical,
            solution_values_numerical,
            label=f"Numerical Solution",
            color="orange",
        )
        plt.xlabel("Time")
        plt.ylabel("Solution")
        plt.title(f"{text} Solution of Riccati problem")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    initial_time = 0.1
    final_time = 0.2
    time_step = 1.0e-3
    integration_time_step = 1.0e-4

    problem = Ricatti()

    integrator = Euler()
    # integrator = RK_second_order()
    # integrator = Merson()

    start_x = problem.get_exact_solution(initial_time)

    numerical_solutions = solve_loop(
        initial_time,
        final_time,
        time_step,
        integration_time_step,
        problem,
        integrator,
        [start_x],
    )
    if not numerical_solutions:
        print("Error: Solution failed.")
        exit(1)

    max_error = 0.0
    l1_error = 0.0
    l2_error = 0.0
    last_t = 0
    diff = -1
    for time, x in numerical_solutions:
        if diff != -1:
            tau = time - last_t
            l1_error += diff * tau
            l2_error += diff * diff * tau
            max_error = max(max_error, diff)
        exact_solution = problem.get_exact_solution(time)
        diff = abs(exact_solution - x[0])

    print("L1 error:", l1_error)
    print("L2 error:", math.sqrt(l2_error))
    print("Max error:", max_error)

    exact_solutions = problem.get_exact_solutions(initial_time, final_time, time_step)
    problem.plot_solution(exact_solutions, numerical_solutions, "Exact vs Numerical")
