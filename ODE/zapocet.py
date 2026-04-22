import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
eps = sys.float_info.epsilon

#iniciace třídy, left right podmínky dodáme při vytvoření
class zapocet:
    def __init__(self):
        self.Lcon = None
        self.Rcon = None
        self.start = math.pi/2
        self.end = math.pi
        self.nb_of_knots = None

    def Net_method(self, Lcon, Rcon, nb_of_knots):
        self.Lcon = Lcon
        self.Rcon = Rcon
        self.nb_of_knots = nb_of_knots

        #vytvoření polí o velikosti sítě
        u = [0.0] * self.nb_of_knots
        alpha = [0.0] * self.nb_of_knots
        beta = [0.0] * self.nb_of_knots
        p = [0.0] * self.nb_of_knots
        q = [0.0] * self.nb_of_knots
        c = [0.0] * self.nb_of_knots
        A = [0.0] * self.nb_of_knots
        B = [0.0] * self.nb_of_knots

        #zpracování alf a bet podle vzorců z přednášky
        alpha[0] = 0.0
        beta[0] = self.Lcon

        #velikost kroku
        h = (self.end - self.start) / (self.nb_of_knots)
        
        #dosazení do typové úlohy
        for i in range (self.nb_of_knots):
            p[i] = math.exp(-math.sin(i*h + math.pi/2))
            q[i] = -math.exp(-math.sin(i*h + math.pi/2))*math.sin(i*h + math.pi/2)

        #dosazení pro thomasův algoritmus dle přednášky
        for i in range (self.nb_of_knots-1):
            c[i] = -((p[i+1] + p[i])/(h*h + eps) + q[i])
            A[i] = -p[i]/(h*h + eps)
            B[i] = -p[i+1]/(h*h + eps)


        for j in range (self.nb_of_knots-1):
            alpha[j+1] = (B[j])/(c[j] - alpha[j]*A[j] + eps)
            beta[j+1] = (beta[j]*A[j])/(c[j] - alpha[j]*A[j] + eps)
        
        #pravý okraj a z něho začínáme zpětně počítat dle thomasova algo
        u[self.nb_of_knots-1] = self.Rcon
        
        for i in range (self.nb_of_knots-2, -1, -1):
            u[i] = alpha[i+1]*u[i+1] + beta[i+1]

        return u
    
    def exact_solution(self):
        u = [0.0] * self.nb_of_knots
        h = (self.end - self.start) / (self.nb_of_knots)
        for i in range (self.nb_of_knots):
            u[i] = math.exp(math.sin(i*h + math.pi/2))
        return u

    def plot_solution(self, u_values, exact_values, t):
        #plot řešení, přímka času a hodnot u
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            t,
            u_values,
            label=f"Numerical Solution",
            color="orange",
        )
        plt.plot(
            t,
            exact_values,
            label = f"Exact Solution",
            color = "blue",
        )
        plt.xlabel("Time")
        plt.ylabel("Solution")
        plt.title(f"Meoda sítí")
        plt.legend()
        plt.grid(True)
        plt.show()

#výpočet konkrétního řešení
if __name__ == "__main__":
    problem = zapocet()
    u_problem = problem.Net_method(math.exp(1), 1, 100)
    # Spojíme je do dvojic (sloupců)
    t = np.linspace(problem.start, problem.end, problem.nb_of_knots)
    data_to_save = np.column_stack((t, u_problem))
    exact_values = problem.exact_solution()
    # Uložíme
    np.savetxt('vysledek.csv', data_to_save, delimiter=',', header='x, y', comments='')
    #plot
    problem.plot_solution(u_problem, exact_values, t)
    