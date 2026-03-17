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

E_1 = 0
E_2 = 0
erros = []
EOC_arr= []
steps = []
integration_time_step_def = 1.0e-2
for i in range(5):
    
    if __name__ == "__main__":
        initial_time = 1 #far from divergence
        final_time = 10
        time_step = 1.0e-1
        integration_time_step = integration_time_step_def/(2**(i))
        steps.append( (final_time - initial_time)/integration_time_step )
        problem = Ricatti()

        start_x = problem.get_exact_solution(initial_time)

        integrator = Euler()
        #integrator = RK_second_order()
        # integrator = Merson()

        

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
        #špatně error asi, možna diff funkci, protože to tam ma taky jinak 
        for time, x in numerical_solutions:
            if diff != -1:
                tau = time - last_t
                l1_error += diff * tau
                l2_error += diff * diff * tau
                max_error = max(max_error, diff)
            exact_solution = problem.get_exact_solution(time)
            diff = abs(exact_solution - x[0])
        l2_error = math.sqrt(l2_error) #odmocnina !!
        erros.append(l2_error)
        E_1 = E_2
        E_2 = l2_error
        if E_1 != 0:
            EOC = (math.log10( (E_1)/ (E_2)))/math.log10( (integration_time_step*2)/(integration_time_step))
            EOC_arr.append(EOC)
        print("L1 error:", l1_error)
        print("L2 error:", math.sqrt(l2_error))
        print("Max error:", max_error)
        print("Initial solution:", start_x)

        exact_solutions = problem.get_exact_solutions(initial_time, final_time, time_step)
        #problem.plot_solution(exact_solutions, numerical_solutions, "Exact vs Numerical")

for i in range(len(erros)):
    print(f"Error for integration time step {integration_time_step_def/(2**(i))}: {erros[i]}")
for i in range(len(EOC_arr)):
    print(f"EOC for integration time step {integration_time_step_def/(2**(i))}, {integration_time_step_def/(2**(i+1))}: {EOC_arr[i]}")
print("Steps:", steps)