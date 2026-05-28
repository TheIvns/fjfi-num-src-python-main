import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time as t
from scipy.sparse import lil_matrix
from Linear_systems.stationary_solver import StationarySolver
from merson import Merson


class HeatEquationProblem2D:
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.hx = 1.0 / (sizeX - 1)
        self.hy = 1.0 / (sizeY - 1)

    def get_degrees_of_freedom(self):
        return self.sizeX * self.sizeY

    def set_initial_condition(self, u, r=0.1):
    # Střed kružnice nastaven na střed oblasti [0.5, 0.5]
        x0 = 0.5
        y0 = 0.5
        
        for i in range(self.sizeX):
            x = i * self.hx
            for j in range(self.sizeY):
                y = j * self.hy
                
                distance_sqr = (x - x0)**2 + (y - y0)**2
                
                if distance_sqr < r**2:
                    u[j, i] = 1.0  
                else:
                    u[j, i] = 0.0
        return u

    def set_initial_condition_from_pgm(self, pgm_file):
        u = np.loadtxt(pgm_file, skiprows=4)  # Skip the first 4 lines of the PGM file
        u = u / np.max(u)  # Normalize the image data
        u = np.flipud(u)  # Flip the image vertically to match the orientation of the grid
        return u

    def function_f(self, time, u, k=None):
        # Přetvarujeme plochý vektor zpět na 2D mřížku (řádky = Y, sloupce = X)
        u = u.reshape((self.sizeY, self.sizeX))
        laplacian = np.zeros_like(u)
        
        # Předpočítáme si jmenovatele pro zrychlení
        hx_sqr = self.hx ** 2
        hy_sqr = self.hy ** 2

        for j in range(1, self.sizeY - 1):
            for i in range(1, self.sizeX - 1):
                # Druhá derivace podle X
                d2u_dx2 = (u[j, i+1] - 2 * u[j, i] + u[j, i-1]) / hx_sqr
                # Druhá derivace podle Y
                d2u_dy2 = (u[j+1, i] - 2 * u[j, i] + u[j-1, i]) / hy_sqr
                
                # Celkový Laplacián v bodě [j, i]
                laplacian[j, i] = d2u_dx2 + d2u_dy2

        # Pro Mersona musíme vrátit jednorozměrné pole
        return laplacian.flatten()
        # todo - calculate laplacian


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
final_time = 0.01
time_step = 0.001
integration_time_step = 0.001
sizeX = 50
sizeY = 50

if __name__ == "__main__":
    problem = HeatEquationProblem2D(sizeX, sizeY)
    u = np.zeros((sizeX, sizeY))
    """pgm_file = "motyl.txt"
    u = problem.set_initial_condition_from_pgm(pgm_file)"""
    u = problem.set_initial_condition(u)
    problem.write_solution(0.0, 0, u)

    stationary = True  

    A = np.eye(sizeX * sizeY, sizeX * sizeY) #aby to nepadalo při inciaci solveru, stejně se to pak přepíše
    b = np.zeros(sizeX * sizeY)


    
    start = t.time()
    if stationary:
        solver = StationarySolver(A, b)
        solver.set_max_iterations(10000)
        solver.relaxation = 1.9
        print("sor par", solver.relaxation)
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
                            # Sousedé v ose X (levý a pravý)
                            index_left  = j * sizeX + (i - 1)  
                            index_right = j * sizeX + (i + 1)  

                            # Sousedé v ose Y (dolní a horní)
                            index_down  = (j - 1) * sizeX + i   
                            index_up    = (j + 1) * sizeX + i   

                            A[index, index_left]  = -lambda_x
                            A[index, index_right] = -lambda_x

                            A[index, index]       = 1.0 + 2.0 * lambda_x + 2.0 * lambda_y

                            A[index, index_down]  = -lambda_y
                            A[index, index_up]    = -lambda_y
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