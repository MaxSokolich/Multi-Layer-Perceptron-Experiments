import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import json
import os
from chop import Chop
import json    




def load_data(data, plot = False):
    initial_configuration = np.array(data['initial_configuration'])  
    all_commands = np.array(data['all_commands'])
    T_values = np.array((data['T_values']))  
     
    actions = []
    for vec in all_commands:
        actions.append(vec)
    
    chop_c = Chop(all_commands, T_values, initial_configuration)
    chop_step = chop_c.get_chop_step()
    path = chop_c.get_chopped_trajectory(chop_step)
    obs_ls = []
    goal = np.array(path[-1])
    N =2
    for i in range(len(path)):
        agents_array = np.asarray(path[i]).flatten()
        goals_array = np.asarray([10, -10.1, 40.1, 35.1]).flatten()
        obs_ls.append(np.hstack((agents_array, goals_array)))
        # obs = {"agents": np.array(path[i]).reshape(2, 2), "goals": goal}
        # obs_ls.append(obs)
    path1 = path[:,0:2]
    path2 = path[:,2:4]
    actions_chopped = actions*chop_step
    actions_chopped = np.array(actions_chopped)

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



    # Wrap in DictObs

    obs_ls = np.array(obs_ls)
    return obs_ls[:-1,:], actions_chopped

        
       
 

folder_path = "/home/mker/ubot_RL/Multi-Layer-Perceptron-Experiments/RL_approach/imitation_data"

all_obs = []
all_actions = []
files = os.listdir(folder_path)
size_data = 10**5
trajectories = []
# Load and parse each file
i = 0 
for file_name in files[:size_data]:
    if file_name.endswith(".json"):
        with open(os.path.join(folder_path, file_name), "r") as f:
            data = json.load(f)
            observations, actions = load_data(data, plot = False)
            all_obs.extend(observations)
            all_actions.extend(actions)
    print(f"Loaded {file_name} ({i+1}/{size_data})")
    i += 1

# Save
all_obs = torch.tensor(all_obs)
all_actions = torch.tensor(all_actions)
torch.save((all_obs, all_actions), "dataset100k.pt")