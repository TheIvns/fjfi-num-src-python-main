import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time as t
from scipy.sparse import lil_matrix
from Linear_systems.stationary_solver import StationarySolver
from Merson import Merson


class HeatEquationProblem2D:
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.hx = 1.0 / (sizeX - 1)
        self.hy = 1.0 / (sizeY - 1)

    def get_degrees_of_freedom(self):
        return self.sizeX * self.sizeY

    def set_initial_condition(self, u, r=0.3):
        # todo set initial conditions
        pass

    def set_initial_condition_from_pgm(self, pgm_file):
        u = np.loadtxt(pgm_file, skiprows=4)  # Skip the first 4 lines of the PGM file
        u = u / np.max(u)  # Normalize the image data
        u = np.flipud(u)  # Flip the image vertically to match the orientation of the grid
        return u

    def function_f(self, time, u, k=None):
        u = u.reshape((self.sizeY, self.sizeX))
        laplacian = np.zeros_like(u)

        # todo - calculate laplacian

        return laplacian.flatten()

    def write_solution(self, t, step, u):
        filename = f"heat-equation-2d-{step:05d}.txt"
        np.savetxt(filename, u)
        return True

    @staticmethod
    def plot_solution(steps, sizeX, sizeY, save_fig=False):
        for step in steps:
            filename = f"heat-equation-2d-{step:05d}.txt"
            u = np.loadtxt(filename).reshape((sizeY, sizeX))
            plt.imshow(u, extent=[0, 1, 0, 1], origin='lower')
            if save_fig:
                plt.savefig(f"butterfly-{step:05d}.png", bbox_inches='tight', format='png')
            plt.show()

initial_time = 0.0
final_time = 0.0001
time_step = 0.00001
integration_time_step = 0.01
sizeX = 434
sizeY = 606

if __name__ == "__main__":
    problem = HeatEquationProblem2D(sizeX, sizeY)

    pgm_file = "motyl.txt"
    u = problem.set_initial_condition_from_pgm(pgm_file)

    problem.write_solution(0.0, 0, u)

    stationary = False

    # Use LIL matrix (list of lists) for ease of modification
    A = lil_matrix((sizeX * sizeY, sizeX * sizeY))
    b = np.zeros(sizeX * sizeY)

    # todo - Set boundary conditions for A

    start = t.time()
    if stationary:
        solver = StationarySolver(A, b)
        solver.set_max_iterations(10000)
        solver.relaxation = 1.6
    else:
        solver = Merson()
        solver.setup(problem.get_degrees_of_freedom())

    time = initial_time
    last_tau = -1.0
    hx_sqr = (1.0 / (sizeX - 1)) ** 2
    hy_sqr = (1.0 / (sizeY - 1)) ** 2
    step = 0
    steps_to_plot = [0]

    while time < final_time:
        stop_time = min(time + time_step, final_time)
        print(f"Time = {time} step = {step}")
        if stationary:
            while time < stop_time:
                current_tau = min(integration_time_step, stop_time - time)
                if current_tau != last_tau:
                    # Set-up lin sys
                    lambda_x = current_tau / hx_sqr
                    lambda_y = current_tau / hy_sqr
                    for j in range(1, sizeY - 1):
                        for i in range(1, sizeX - 1):
                            index = j * sizeX + i
                            # todo - update A
                # right-hand side b
                b[:] = u.flatten()
                # Solve the lin system using SOR
                print("sor")
                success = solver.solve(method="sor")
                if success is False:
                    exit("Solver failed!")
                u = solver.iteration_results[-1].reshape((sizeY, sizeX))
                time += current_tau
                last_tau = current_tau
        else:
            # Merson
            time, u_flat, success = solver.solve(integration_time_step, stop_time, time, problem, u.flatten())
            u = u_flat.reshape((sizeY, sizeX))
        step += 1
        steps_to_plot.append(step)
        problem.write_solution(time, step, u)

    stop = t.time()
    print(f"The time: {stop - start:.2f} seconds")

    problem.plot_solution(steps_to_plot, sizeX, sizeY, save_fig=True)
