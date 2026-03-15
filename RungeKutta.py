import numpy as np

class RK_second_order:
    def __init__(self):
        self.dof = 0
        self.k1 = None
        self.k2 = None
    def setup(self, degrees_of_freedom):
        self.k1 = np.zeros(degrees_of_freedom)
        self.k2 = np.zeros(degrees_of_freedom)
    def solve(self, integration_time_step, stop_time, time, problem, x):
        iteration = 0
        while time < stop_time:
            tau = min(integration_time_step, stop_time - time)

            # todo - implement changes for k1, k2 and x
            #hw viz slide
            time += tau
            iteration += 1
        return time, x, True