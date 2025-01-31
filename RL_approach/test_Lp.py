import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Load data
T = np.load('time_periods.npy')
commands = np.load('commands.npy')
given_position = np.array([1.52, 1.1, -1.1, -1.1])
initial_configuration = np.array([1.0, 1.1 , -1.1 , -1.1])
# Get the size of T
size_T = len(commands)

# Create a Gurobi model
m = gp.Model()

# Add decision variables for the time periods
W = m.addMVar(size_T, ub=T.flatten(), lb=0, name='Time periods')

# Calculate the weighted sum
weighted_sum = commands.T @ W +initial_configuration

# Add auxiliary variables for absolute differences
abs_diff = m.addMVar(len(given_position), lb=0, name='Absolute differences')

# Add constraints to linearize absolute value
for i in range(len(given_position)):
    m.addConstr(abs_diff[i] >= given_position[i] - weighted_sum[i])
    m.addConstr(abs_diff[i] >= weighted_sum[i] - given_position[i])

# Set the objective to minimize the sum of absolute differences
m.setObjective(abs_diff.sum(), GRB.MINIMIZE)
m.update()
# Optimize the model
m.optimize()

# Check if optimization was successful and print results
if m.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    print("Runtime", m.Runtime)
    print("Objective value:", m.ObjVal)
else:
    print("Optimization was not successful.")
