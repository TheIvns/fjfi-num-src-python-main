import numpy as np
import sys


class Euler:
    def __init__(self):
        self.k1 = None

    def setup(self, degrees_of_freedom):
        self.k1 = np.zeros(degrees_of_freedom)

    def solve(self, integration_time_step, stop_time, time, problem, x):
        iteration = 0
        while time < stop_time:
            tau = min(integration_time_step, stop_time - time)
            self.k1 = problem.function_f(time, x)
            x += tau * self.k1
            time += tau
            iteration += 1
        return time, x, True
