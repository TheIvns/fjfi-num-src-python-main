import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import math

sys.path.append("..")
from euler import Euler
from ODE.ODE import *

initial_time = 0.0
final_time = 0.0001
time_step = 0.0000001
integration_time_step = 0.0000001
L = 1.0  # Délka delší strany
eps = sys.float_info.epsilon

class ExplicitHeatEquationProblem2D:
    def __init__(self, width, height, L):
        self.width = width
        self.height = height
        
        self.h = L / (max(width, height) - 1)

    def get_degrees_of_freedom(self):
        return self.width * self.height

    def set_initial_condition(self, u, img): 
        if img is not None:
            u[:] = img
        else:
            u.fill(0.0) 

    def function_f(self, t, u):
        fu = np.zeros_like(u)
        
        fu[1:-1, 1:-1] = (
            (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / (self.h**2 + eps) +
            (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / (self.h**2 + eps)
        )
        return fu

if __name__ == "__main__":
    try:
        img_flat = np.loadtxt("motyl.txt")
        print(f"Počet načtených hodnot: {len(img_flat)}")
    except:
        print("Soubor motyl.txt nenalezen, generuji náhodná data pro test.")
        img_flat = np.random.rand(434 * 606) * 255

    width, height = 434, 606 
    total_needed = width * height
    
    if len(img_flat) < total_needed:
        print(f"Varování: Málo dat ({len(img_flat)}), doplňuji zbytek bílou.")
        temp = np.full(total_needed, 255.0)
        temp[:len(img_flat)] = img_flat
        img_flat = temp
    
    img_final = img_flat[:total_needed].reshape((height, width)) / 255.0

    problem = ExplicitHeatEquationProblem2D(width, height, L)
    #vyhnu se set initial con
    u_init = np.copy(img_final)
    global_M = np.max(u_init)


    integrator = Euler()
    global_solution = solve_loop(
        initial_time,
        final_time,
        time_step,
        integration_time_step,
        problem,
        integrator,
        u_init,
    )

    t_list, u_list = zip(*global_solution)
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    cax = ax1.imshow(u_list[0], cmap="hot", origin="lower")
    ax1.set_title("Heatmap (Diffusion)")
    fig.colorbar(cax, ax=ax1)

    ax2 = fig.add_subplot(122, projection="3d")
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    def animate(i):
        ax2.clear()
        cax.set_array(u_list[i])
        surf = ax2.plot_surface(X, Y, u_list[i], cmap="hot", antialiased=False)
        ax2.set_zlim(0, 1.1 * global_M)
        ax2.set_title(f"Time: {t_list[i]:.7f}")
        return cax,

    ani = animation.FuncAnimation(fig, animate, frames=len(t_list), interval=50, blit=False)
    plt.tight_layout()
    plt.show()