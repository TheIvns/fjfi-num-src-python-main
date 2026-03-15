import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import sys

sys.path.append("..")

from RungeKutta import RK_second_order
from Euler import Euler
from Merson import Merson
from ODE import solve_loop

particles_count = 10
initialTime = 0.0
finalTime = 100.0
timeStep = 0.1
integrationTimeStep = 1.0e-1
G = 1.0
epsilon = 0.01
trace_length = 10

global_solution = None
global_masses = None
position_history = np.zeros((trace_length, particles_count, 3))


def animate(frame):
    t, u = zip(*global_solution)
    frame_idx = int(frame % (finalTime - initialTime) / timeStep)
    try:
        u = np.reshape(u[frame_idx], (2 * particles_count, 3))
    except:
        print("EXIT")
        exit()
    positions = u[:particles_count]

    # Update position history (shift older positions)
    position_history[:-1] = position_history[1:]  # Shift all previous positions
    # print(position_history[-1])
    position_history[-1] = positions  # Add new positions

    # Update scatter plot positions
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    # Update trace lines
    for i in range(particles_count):
        traces[i].set_data(position_history[:, i, 0], position_history[:, i, 1])
        traces[i].set_3d_properties(position_history[:, i, 2])

    return scat, *traces


class NBodyProblem:
    def __init__(self):
        self.particles_count = 10
        self.G = 1.0
        self.epsilon = 0.1
        self.masses = np.zeros(self.particles_count)

    def setParameters(self, _particles_count, _G, _epsilon):
        self.particles_count = _particles_count
        self.G = _G
        self.epsilon = _epsilon
        self.masses = np.zeros(self.particles_count)

    def setMasses(self, _masses):
        self.masses = _masses

    def get_degrees_of_freedom(self):
        return 6 * self.particles_count

    def function_f(self, t, _u):
        # TODO: Implement nbody problem
        fu = np.random.rand(np.shape(_u)[0], np.shape(_u)[1]) * 5 - 2.5 - 0.05 * _u
        return fu


if __name__ == "__main__":

    problem = NBodyProblem()
    problem.setParameters(particles_count, G, epsilon)

    # np.random.seed(42)  # For reproducibility
    u = np.zeros(6 * particles_count)
    u = np.reshape(u, (2 * particles_count, 3))
    positions = np.random.rand(particles_count, 3) * 20 - 10
    velocities = np.random.randn(particles_count, 3) * 0.5
    global_masses = np.random.rand(particles_count) * 2 + 1

    problem.setMasses(global_masses)
    u[:particles_count] = positions
    u[particles_count:] = velocities
    u = np.reshape(u, (1, 6 * particles_count))

    integrator = Euler()

    try:
        global_solution = solve_loop(
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

    # Set up figure for 3D animation
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    # Create scatter plot for particles
    scat = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=global_masses * 20,
        c="blue",
        alpha=0.7,
    )

    # Create line objects for traces
    traces = [ax.plot([], [], [], "r-", alpha=0.5)[0] for _ in range(particles_count)]

    # Create 3D animation
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=10, blit=False)

    plt.show()
