import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import sys
import json
import scipy.io as sio
import os
from chop import Chop
import json    


def load_data(data, plot = False):
   
       
    initial_configuration = np.array(data['initial_configuration'])  
    all_commands = np.array(data['all_commands'])
    T_values = np.array((data['T_values']))  
     

    
    chop_c = Chop(all_commands, T_values, initial_configuration)
    chop_step = chop_c.get_chop_step()
    path = chop_c.get_chopped_trajectory(chop_step)
    path1 = path[:,0:2]
    path2 = path[:,2:4]
    if plot:
        fig, ax = plt.subplots()
        ax.plot(path1[:,0], path1[:,1], 'o-', color = 'blue')
        ax.plot(path2[:,0], path2[:,1], 'o-', color = 'red')
        ax.plot(initial_configuration[0], initial_configuration[1], 'go')
        ax.plot(initial_configuration[2], initial_configuration[3], 'go')
        ax.plot(path1[-1][0], path1[-1][1], 'o', color = 'black')
        ax.plot(path2[-1][0], path2[-1][1], 'o', color = 'black')
        # ax.set_aspect('equal')
        ax.legend(['robot1', 'robot2', 'initial1', 'initial2','final1', 'final2'])
        plt.show()

        
       
 


# initial_configuration = np.array([  1.1, 1.1 , -1.1, -1.1])   

# all_commands, frq_map = get_all_commands()
# all_commands, T_values = LP(initial_configuration, final_configuration, all_commands, frq_map, plot=True)

with open('imitation_data/lp_result_4166667.json', 'r') as f:
    data = json.load(f)

# Now `data` is a Python dictionary (or list, depending on the JSON structure)
load_data(data, plot = True)