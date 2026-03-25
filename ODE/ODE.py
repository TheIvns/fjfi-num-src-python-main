import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def solve_loop(initial_time, final_time, time_step, integration_time_step, problem, solver, x):
    solver.setup(problem.get_degrees_of_freedom())
    time_steps_count = math.ceil(max(0.0, final_time - initial_time) / time_step)
    time = initial_time
    solution = [(time, np.array(x))]  # Store the initial solution

#tqdm is used to show a progress bar for the loop
    for step in tqdm(range(1, time_steps_count + 1), desc="Getting numerical solution"):
        current_stop_time = time + min(time_step, final_time - time)
        time, x, success = solver.solve(integration_time_step, current_stop_time, time, problem, x)
        if not success:
            return False, 0.0, 0.0, 0.0  # Return error if solution fails
        solution.append((time, np.array(x)))  # Need to make a copy of x
    return solution
