import numpy as np
import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt

import sys
import json
import scipy.io as sio
import os
from chop import Chop
import json    
import imitation
from imitation.data.types import Trajectory, DictObs
# from imitation.data.types import flatten_trajectories
from imitation.data.types import Transitions
import pickle


def convert_to_dictobs(traj):
    obs_dicts = []
    for obs in traj.obs:
        agents = np.asarray(obs["agents"], dtype=np.float32)
        goals = np.asarray(obs["goals"], dtype=np.float32)
        obs_dict = DictObs({"agents": agents, "goals": goals})
        obs_dicts.append(obs_dict)

    return Trajectory(
        obs=obs_dicts,
        acts=traj.acts,
        infos=traj.infos,
        terminal=traj.terminal
    )

def make_trajectory(obs_raw, acts, terminal=True):
    obs_dicts = []

    N = 2
    goal = obs_raw[-1]
    for obs in obs_raw:
            obs_dicts.append({
                "agents": np.array(obs[:2*N]).reshape(N, 2),
                "goals": goal 
            })

    traj = Trajectory(obs=obs_dicts, acts=acts, terminal=terminal)

    return traj





def decode_action(vec, n_angles=4):
    lookup_table = [[
                2.72, 4.06, 5.80, 6.81, 9.07, 9.46, 11.32, 11.95, 14.11, 14.49,
                16.15, 16.49, 17.30, 17.09, 18.35, 19.68, 19.45, 21.39, 22.50,
                23.65
            ],
            [
                16.62, 27.30, 37.71, 48.13, 58.72, 66.67, 78.15,
                84.48, 96.43, 108.05, 119.22, 120.53, 127.00,
                133.90, 131.50, 151.17, 153.06, 161.49, 170.00,
                170.95
            ]]
    # Extract magnitudes and reconstruct angle
    mag0 = np.linalg.norm(vec[:2])
    mag1 = np.linalg.norm(vec[2:])
    
    # Estimate angle from any pair
    alpha = np.arctan2(vec[1], vec[0])  # atan2(y, x)

    # Wrap angle to [0, 2π]
    if alpha < 0:
        alpha += 2 * np.pi

    # Quantize angle to nearest bucket
    dt_alpha = 2 * np.pi / n_angles
    j = int(np.round(alpha / dt_alpha)) % n_angles

    # Find the closest matching (mag0, mag1) in the lookup table
    distances = [np.linalg.norm(np.array([mag0, mag1]) - np.array([lookup_table[0][idx], lookup_table[1][idx]])) for idx in range(len(lookup_table))]
    i = int(np.argmin(distances))

    # Compute discrete action index
    action_index = i * n_angles + j
    return action_index


def load_data(data, plot = False):
   
       
    initial_configuration = np.array(data['initial_configuration'])  
    all_commands = np.array(data['all_commands'])
    T_values = np.array((data['T_values']))  
     
    actions = []
    for vec in all_commands:
        actions.append(decode_action(vec))
    
    chop_c = Chop(all_commands, T_values, initial_configuration)
    chop_step = chop_c.get_chop_step()
    path = chop_c.get_chopped_trajectory(chop_step)
    obs_ls = []
    goal = np.array(path[-1])
    N =2
    for i in range(len(path)):
        agents_array = np.asarray(path[i], dtype=np.float32).reshape(N, 2)
        goals_array = np.asarray([10, -10.1, 40.1, 35.1], dtype=np.float32).reshape(N, 2)
        obs_ls.append({"agents": agents_array, "goals": goals_array})
        # obs = {"agents": np.array(path[i]).reshape(2, 2), "goals": goal}
        # obs_ls.append(obs)
    path1 = path[:,0:2]
    path2 = path[:,2:4]
    actions_chopped = actions*chop_step
    actions_chopped = np.array(actions_chopped)
    # for n in range(chop_step):
    #     actions_chopped.append(actions)
    # actions_chopped = np.array(actions_chopped).flatten()

    
    # actions = np.reshape(actions, (-1,4))
    # print(actions.shape)
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

    agents_stack = np.stack([np.array(obs["agents"], dtype=np.float32) for obs in obs_ls])
    goals_stack = np.stack([np.array(obs["goals"], dtype=np.float32) for obs in obs_ls])

    # Wrap in DictObs
    batched_obs = DictObs({"agents": agents_stack, "goals": goals_stack})

    return batched_obs, actions_chopped

        
       
 


# with open('/home/mker/ubot_RL/Multi-Layer-Perceptron-Experiments/RL_approach/imitation_data/lp_result_4166667.json', 'r') as f:
#     data = json.load(f)

# # Now `data` is a Python dictionary (or list, depending on the JSON structure)
# observation, actions = load_data(data, plot = True)

folder_path = "/home/mker/ubot_RL/Multi-Layer-Perceptron-Experiments/RL_approach/imitation_data"

all_obs = []
all_actions = []
files = os.listdir(folder_path)
size_data = 10**5
trajectories = []
# Load and parse each file
for file_name in files[:size_data]:
    if file_name.endswith(".json"):
        with open(os.path.join(folder_path, file_name), "r") as f:
            data = json.load(f)
            observations, actions = load_data(data, plot = False)
       
            # acts = actions
            # dones = np.zeros(len(obs), dtype=bool)
            # dones[-1] = True  # Mark last step as terminal

            trajectories.append(Trajectory(obs=observations, acts=actions, terminal=True, infos=None))


            # trajectories.a`ppend(Trajectory(obs=observations, acts=actions, terminal=True, infos=[{}]*len(actions)))

# # transitions = flatten_trajectories(trajectories)
# Save
with open("trajectories_100k.pkl", "wb") as f:
    pickle.dump(trajectories, f)


# converted_trajs = [convert_to_dictobs(traj) for traj in trajectories]

# from stable_baselines3.common.env_util import make_vec_env
# import pickle
# from imitation.algorithms.bc import BC
# from ubots_env import *
# from gym.wrappers import FlattenObservation
# # env = make_vec_env("YourEnvName-v0", n_envs=1)  # 
# def make_single_env(env_kwargs):

#     def _init():
       
#         env = uBotsGymDiscrete(N=2, **env_kwargs)
#         return env

#     return _init
# # with open("transitions.pkl", "rb") as f:
# #     transitions = pickle.load(f)


# env_kwargs = dict(XMIN=-100,
#                  XMAX=100,
#                  YMIN=-100,
#                  YMAX=100,
#                  horizon=200
#                 )
    
# env = make_vec_env(make_single_env(env_kwargs), n_envs=1)


# rng = np.random.default_rng(seed=0)  # 

# bc_trainer = BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=trajectories,
#     batch_size=64,
#     ent_weight=0.0,
#     l2_weight=1e-5,  # NEW: directly use this
#     rng=rng  # required in imitation 1.0+
# )


# bc_trainer.train(n_epochs=20) 
# bc_trainer.policy.save("bc_policy_ubots.zip")
