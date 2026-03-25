import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

sys.path.append("..")
from RungeKutta import RK_second_order
from euler import Euler
from merson import Merson
from ODE import solve_loop


class VolterraLotkaProblem:
    def __init__(self):
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        self.d = 1.0


    def setParameters(self, _a, _b, _c, _d):
        self.a = _a
        self.b = _b
        self.c = _c
        self.d = _d

    def get_degrees_of_freedom(self):
        return 2

    def function_f(self, t, _u):
        u1, u2 = _u
        fu = np.zeros(2)
        fu[0] = self.a * u1 - self.b * u1 * u2
        fu[1] = -self.c * u2 + self.d * u1 * u2
        return fu

    def plot_solution(self, solution):
        t, data = zip(*solution)
        u = np.asarray(data)
        u1 = u[:, 0]
        u2 = u[:, 1]

        print("Initial condition: t:", t[0], " U_1:", u1[0], " U_2:", u2[0])

        

        plt.plot(t, u1, label="u_1")
        plt.plot(t, u2, label="u_2")
        plt.xlabel("Time")
        plt.ylabel("Solution")
        plt.title("Volterra-Lotka Problem Solution")
        plt.legend()
        plt.grid(True)
        plt.show()

for i in range(5):

    if __name__ == "__main__":
        initialTime = 0.0
        finalTime = 20
        timeStep = 1.0e-2
        integrationTimeStep = 1.0e-3

        problem = VolterraLotkaProblem()
        problem.setParameters(1.0 + i, 1.0, 1.0, 1.0)

        integrator = Euler()

        u = np.array([1.0, 0.5])

        try:
            solution = solve_loop(
                initialTime,
                finalTime,
                timeStep,
                integrationTimeStep,
                problem,
                integrator,
                u,
            )
            
        


        except Exception as e:
            print(e)
            print("EXIT")

        problem.plot_solution(solution)
