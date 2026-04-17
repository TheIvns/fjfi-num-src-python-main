import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import sys
from os.path import dirname

sys.path.append("..")

from euler import Euler
from merson import Merson
from RungeKutta import RK_second_order
from ODE import *


class SIRModel:
    def __init__(self, N, n, f, b, c, m, mI):
        self.N = N  # total population
        self.n = n  # rate of new susceptible individuals (birth rate)
        self.f = f  # fraction of individuals becoming immune without infection
        self.b = b  # transmission rate
        self.c = c  # recovery rate
        self.m = m  # mortality rate
        self.mI = mI  # mortality rate for infected people

    def get_degrees_of_freedom(self):
        return 3

    def function_f(self, t, u, params=None):
        S, I, R = u
        dS_dt = self.n * (1 - self.f) * self.N - (self.b * I * S) / self.N - self.m * S
        dI_dt = (self.b * I * S) / self.N - self.c * I - self.mI * I
        dR_dt = self.c * I + self.n * self.f * self.N - self.m * R
        return np.array([dS_dt, dI_dt, dR_dt])

    def plot(self, solution):
        t, data = zip(*solution)
        S, I, R = [], [], []
        for d in data:
            S.append(d[0])
            I.append(d[1])
            R.append(d[2])
        I = np.array(I)
        S = np.array(S)
        R = np.array(R)
        plt.plot(t, S, label="Susceptible")
        plt.plot(t, I, label="Infected")
        end_of_infection = np.where(I <= 0.0001)[0]
        if end_of_infection.size > 0:
            plt.scatter(t[end_of_infection[0]], 0, color="r", label="Infection End")
            print("infection ended")
        plt.plot(t, R, label="Recovered")
        plt.plot(t, S + I + R, label="Population")
        plt.xlabel("Time")
        plt.ylabel("Population")
        model_name = "SIR Model"
        plt.title(model_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(model_name + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()


# Parameters of the model
n = n = 0.000028# birth rate
f = 0.15 # fraction of individuals becoming immune without infection
b = 1.5 # transmission rate   # alfa = 0.208  # covid Iran
c = 0.14  # recovery rate  # beta = 0.085
m = 0.000032  # mortality rate
mi = 0.001 # mortality rate for infected individuals
t_min = 0.0  # initial time
t_max = 200  # end time

time_step = 0.01  # time step
integration_time_step = 0.01
integrator = Merson()
# Initial conditions
S = 0.993
I = 0.007
R = 0
N = S + I + R  # population
# b = b * N
initial_conditions = (S, I, R)  # SIR

sir_model = SIRModel(N, n, f, b, c, m, mi)
solution = solve_loop(
    t_min,
    t_max,
    time_step,
    integration_time_step,
    sir_model,
    integrator,
    initial_conditions,
)
sir_model.plot(solution)
