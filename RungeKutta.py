import numpy as np

class RK_second_order:
    def __init__(self):
        self.k1 = None
        self.k2 = None
        
    
    def setup(self, dof):
        self.k1 = np.zeros(dof)
        self.k2 = np.zeros(dof)
    
    def solve(self, integration_time_step, stop_time, time, problem, x):
        iteration = 0
        while time < stop_time:
            tau = min(integration_time_step, stop_time - time)
            self.k1 = tau*problem.function_f(time, x)
            self.k2 = tau*problem.function_f(time + 0.5*tau, x +0.5*self.k1)
            x += self.k2
            
            time += tau
            iteration += 1
        return time, x, True