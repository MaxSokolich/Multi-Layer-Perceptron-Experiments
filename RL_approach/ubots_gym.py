from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import torch as th
import numpy as np
import matplotlib.pyplot as plt
# import gurobipy as gp
# from gurobipy import GRB
import sys
import scipy.io as sio
import os

import gymnasium as gym
from gymnasium.spaces import Box

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback

from networks.mlp_ppo import CustomActorCriticPolicy



 
def minimum_distance_to_path(midpoints, point):
    def point_to_segment_dist(p1, p2, q):
        v = p2 - p1
        w = q - p1
        c1 = np.dot(w, v)
        if c1 <= 0:  # Closest to p1
            return np.linalg.norm(q - p1)
        c2 = np.dot(v, v)
        if c1 >= c2:  # Closest to p2
            return np.linalg.norm(q - p2)
        t = c1 / c2
        projection = p1 + t * v
        return np.linalg.norm(q - projection)
    
    point = np.array(point)
    midpoints = np.array(midpoints)
    min_distance = float('inf')
    
    for i in range(len(midpoints) - 1):
        p1 = midpoints[i]
        p2 = midpoints[i + 1]
        dist = point_to_segment_dist(p1, p2, point)
        min_distance = min(min_distance, dist)
    
    return min_distance


class uBotsGym(gym.Env):
    """
    Class for creating uBots Gym(nasium) environment.
    Can be trained with Stable-Baselines3.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    LOOKUP_TABLE = [[
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

    def __init__(self,
                 N, # number of uBots
                 XMIN=-100, # min x-coord
                 XMAX=100, # max x-coord
                 YMIN=-100, # min y-coord
                 YMAX=100, # max y-coord
                 dt=0.1, # sampling time
                 horizon=100, # task/episode horizon
                 continuous_task=True, # whether to terminate after reaching goal or time elapsed
                 render_mode=None):
        self.N = N
        self.XMIN = XMIN
        self.XMAX = XMAX
        self.YMIN = YMIN
        self.YMAX = YMAX
        self.dt = dt
        self.expert_trj = np.load('postions.npy')
        self.horizon = horizon
        self.continuous_task = continuous_task
        self.render_mode = render_mode
        vmax  = np.max(np.array(self.LOOKUP_TABLE).flatten())
        # Set observation and action spaces
        self.observation_space = Box(
            low=np.array([[XMIN, YMIN], [XMIN, YMIN], [XMIN, YMIN], [XMIN, YMIN]]),  # Lower bounds for (x, y) of each robot
            high=np.array([[XMAX, YMAX], [XMAX, YMAX], [XMAX, YMAX], [XMAX, YMAX]]),  # Upper bounds for (x, y) of each robot
            shape=(int(2*N), 2),
            dtype=np.float32)
        


        # self.observation_space = Box(
        #     low=np.array([[XMIN, YMIN], [XMIN, YMIN], [0,0], [0,0]]),  # Lower bounds for (x, y) of each robot
        #     high=np.array([[XMAX, YMAX], [XMAX, YMAX], [vmax, vmax], [vmax, vmax]]),  # Upper bounds for (x, y) of each robot
        #     shape=(int(2*N), 2),
        #     dtype=np.float32)
        
        self.action_space = Box(low=np.array([0, -np.pi]),
                                    high=np.array([19, np.pi]), shape=(2,), dtype=np.float32)
        # self.action_space = Box(low=np.array([0, 0]),
                                    # high=np.array([1, 1]), shape=(2,), dtype=np.float32)

        # Create matplotlib figure if rendering
        # self.gen_discrete_commands_LP()
        if render_mode == "human":
            self.fig, self.ax = plt.subplots()
            
    # def gen_discrete_commands_LP(self):

    #     # positions1 = np.load('postions.npy')
    #     freqs = np.linspace(1, 20, 20)
    #     alphas = np.arange(0, 2 * np.pi, np.pi / 16)



    #     nsize = len(self.LOOKUP_TABLE[0])
    #     all_commands = []
    #     for i in range(nsize):
    #         speed1 = self.LOOKUP_TABLE[0][i]
    #         speed2 = self.LOOKUP_TABLE[1][i]
    #         for ia in range(len(alphas)):
    #             v = np.array([
    #                 speed1 * np.cos(alphas[ia]), speed1 * np.sin(alphas[ia]),
    #                 speed2 * np.cos(alphas[ia]), speed2 * np.sin(alphas[ia])
    #             ])
    #             all_commands.append(v)

    #     self.all_commands = np.array(all_commands)


    # def LP_solver(self, initial_configuration):


    #     final_configuration = np.array(self._get_goal()).flatten()
    #     size_T = len(self.all_commands)
    #     m = gp.Model()
    #     T = m.addMVar(size_T,ub = 100, lb = 0, name= 'Time periods')
        
    #     b = m.addVar(lb =0, ub= 10**6)
    #     bounds = m.addMVar(4, lb=0, name="bounds")  # Bounds for constraints
    #     for i in range(4):
    #         m.addConstr(bounds[i] == b) 
    #     max_abs_goal = 0.1
    #     error_margin = m.addMVar(4,ub = max_abs_goal, lb = -max_abs_goal, name= 'error_margin')


    #     abs_error = m.addMVar(4, lb=0, name="abs_error")

    #     # Add constraints to represent absolute value
    #     for i in range(4):
    #         m.addConstr(abs_error[i] >= error_margin[i])   # abs_error >= error_margin
    #         m.addConstr(abs_error[i] >= -error_margin[i])  # abs_error >= -error_margin


    #     m.addConstr(initial_configuration+self.all_commands.T@T == final_configuration+error_margin)

    #     for i in range(1,size_T):
    #         m.addConstr(initial_configuration+self.all_commands.T[:,:i]@T[:i] <= bounds)
    #         m.addConstr(-bounds<=initial_configuration+self.all_commands.T[:,:i]@T[:i] )

    #     cost = gp.quicksum(np.linalg.norm(self.all_commands[i])*T[i] for i in range(size_T))
    #     cost = 10*b
    #     cost += gp.quicksum(T)
    #     cost += gp.quicksum(1*abs_error)
    #     for i, v in enumerate(self.all_commands):
    #         m.addConstr(np.linalg.norm(v)*T[i] <= 500)
        
    #     m.update()
    #     m.setObjective(cost, sense=gp.GRB.MINIMIZE)



    #     m.update()
    #     # m.params.NonConvex = 2
    #     m.optimize()

    #     T_values = T.X  # This is your solution for time periods

    #     dominated_command_arg = np.argmax(T_values)
    #     dominated_command = self.all_commands[dominated_command_arg]
    #     dominated_command_speed = np.linalg.norm(dominated_command[0:2])
    #     dominated_command_speed1 = np.linalg.norm(dominated_command[2:4])
    #     dominated_command_angle = np.arctan2(dominated_command[1], dominated_command[0])
    #     speed_arg = np.argmin(np.abs(self.LOOKUP_TABLE[0]-dominated_command_speed))
    #     return speed_arg, dominated_command_angle
                
   
            

    def check_in_bounds(self):
        #### check if the robots are within the bounds with some margin
        margin = 0.5
        out = np.all(self.positions[:, 0] > self.XMIN + margin) and np.all(self.positions[:, 0] < self.XMAX - margin) and np.all(self.positions[:, 1] > self.YMIN + margin) and np.all(self.positions[:, 1] < self.YMAX - margin)
        return out
    
    def min_distance_to_boundary(self):
        min_distance = np.inf
        for pos in self.positions:
            min_distance = min(min_distance, pos[0] - self.XMIN, self.XMAX - pos[0], pos[1] - self.YMIN, self.YMAX - pos[1])
        return min_distance

    def reset(self, seed=None):
        # Set random seed
        self.observation_space.seed(seed)

        # Generate goal location at start of every episode
        self.goal0_pos, self.goal1_pos = self._get_goal()

        self._steps_elapsed = 0 # for checking horizon

        # create initial robot locations
        self.positions = np.array(self._get_init_robot_pos())
        
        obs = deepcopy(self.positions)
        
        # Add robot speeds to observation numpy array
        # obs = np.append(obs, np.zeros((self.N, 2)), axis=0)
        obs = np.append(obs, np.array([self.goal0_pos, self.goal1_pos]), axis=0)
        # print("Initial positions: ", obs)
        
        

        info = {'horizon': self.horizon, 'is_success': False}

        if self.render_mode == "human":
            # setup the display/render
            self.ax.cla()
            # self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.XMIN, self.XMAX)
            self.ax.set_ylim(self.YMIN, self.YMAX)

            # show the goal positions
            self.scat = self.ax.scatter(self.goal0_pos[0],
                                        self.goal0_pos[1],
                                        c='r')
            self.scat = self.ax.scatter(self.goal1_pos[0],
                                        self.goal1_pos[1],
                                        c='g')

            # show the robot positions
            self.scat = self.ax.scatter(self.positions[:, 0],
                                        self.positions[:, 1],
                                        c='b')

        
      
        # print(f"obs_reset: {obs}", 'shape:', obs.shape)
      
        # print("info: ", info)
        return obs, info

    def step(self, action):
        # the action[0] is between 0 and 1 but I want to map it to an index between 0 and 19
        # action[0] = np.floor(action[0] * 19)
        # action[1] = action[1] * 2 * np.pi - np.pi
        print(f"action: {action}", 'shape:', action.shape)

        f, alpha = action
        new_positions = []
        
        speeds = self.v_i(f)
        # old_positions = deepcopy(self.positions)
        for i, pos in enumerate(self.positions):
            dx = speeds[i] * self.dt * np.cos(alpha)
            dy = speeds[i] * self.dt * np.sin(alpha)
            new_pos = pos + np.array([dx, dy])
            new_pos[0] = np.clip(new_pos[0], self.XMIN, self.XMAX)
            new_pos[1] = np.clip(new_pos[1], self.YMIN, self.YMAX)
            new_positions.append(new_pos)
        self.positions = np.array(new_positions)

        self._steps_elapsed += 1
        obs = deepcopy(self.positions)
        vel = np.array([speeds * np.array([np.cos(alpha), np.sin(alpha)]) for speeds in speeds])
        # obs = np.append(obs, np.array(vel), axis=0)  
        print(f"obs: {obs}", 'shape:', obs.shape)
        obs = np.append(obs, np.array([self.goal0_pos, self.goal1_pos]), axis=0)
        # print("New positions: ", obs)
        # obs = np.array(obs).flatten()
        if not self.check_in_bounds():
            reward = -50.0
            terminated = True
            truncated = True if (self._steps_elapsed >= self.horizon) else False
            info = {'is_success': False, 'n_successes': 0}
        else:

            # Get reward and number of robots successfully reached their goals
            reward, successes = self._get_reward(obs)

            if self.continuous_task:
                terminated = False
            else:
                terminated = successes >= 2
            
            info = {'is_success': successes >= 2, 'n_successes': successes}
            
            truncated = True if (self._steps_elapsed >= self.horizon) else False
            
       
       
        
        # print("info: ", info)
        return obs, reward, terminated, truncated, info

    def render(self):
        self.scat.set_offsets(self.positions)
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.1)

    def close(self):
        plt.close()

    def _get_reward(self, obs,eps=1.0):
        """
        Calculate the rewards for current state.

        Parameters:
            obs: current observation
            eps: threshold for checking goal reach. Default: 1.0

        Returns:
            reward: the reward as a function of distance to goals
            successes: number of robots that successfully reached their corresponding goals            
        """
        rob0_pos = obs[0]
        rob1_pos = obs[1]

        # Calculate dist(robot, goal) for each robot
        d0 = np.linalg.norm(rob0_pos - self.goal0_pos)
        d1 = np.linalg.norm(rob1_pos - self.goal1_pos)

        # Check goal-reach condition
        # successes = sum(np.array([d0, d1]) <= eps)
        # and the array
        successes = (np.array([d0, d1]) <= eps)
        end_goal = successes.all()
        end_goal_one_robot = successes.any()
        
        # Calculate rewards
        # reward = -10.0 * (d0 + d1) + successes
        # expert_command = self.LP_solver(np.array([rob0_pos,rob1_pos]).flatten())
        # alpha_similar = -np.abs(alpha - expert_command[1])
        # freq_similarity = -np.linalg.norm(self.LOOKUP_TABLE[0][int(arg)] - self.LOOKUP_TABLE[0][int(expert_command[0])])
        # -np.linalg.norm(self.LOOKUP_TABLE[1][int(arg)] - self.LOOKUP_TABLE[1][int(expert_command[0])])
        minimum_distance_to_boundary = self.min_distance_to_boundary()
        penalty_to_boundry = -20*np.exp(-0.1*minimum_distance_to_boundary)
        reward = 80*end_goal -8*minimum_distance_to_path(self.expert_trj,np.array([rob0_pos, rob1_pos]).flatten())+penalty_to_boundry
        # reward = -1.0 * (np.exp(d0) + np.exp(d1))        
        # reward = (1.0 - np.tanh(d0)) + (1.0 - np.tanh(d1))
        print(f"Reward: {reward}")

        return reward, successes
    
    def _get_goal(self):
        # Random goal
        # goal0, goal1 = np.random.uniform([5, 5], [self.XMAX, self.YMAX], (self.N, 2))
        

        # Fixed goal
        goal0 = np.array([1.1, 12.1])
        goal1 = np.array([30.1, 35.1])

        return goal0, goal1
    
    def _get_init_robot_pos(self):
        # Random goal
        # rob0_pos, rob1_pos = np.random.uniform([self.XMIN, self.YMIN], [self.XMAX, self.YMAX], (self.N, 2))
        

        # Fixed positions
        rob0_pos = np.array([1.1, 1.1])
        rob1_pos = np.array([-10.1, -1.1])

        return rob0_pos, rob1_pos

    def v_i(self, f):
        if self.N > 2:
            print("Warning: Number of bots is greater than 2. Replicating the lookup table for the first 2 bots.")
            self.LOOKUP_TABLE = self.LOOKUP_TABLE * (self.N // 2 + 1)
        return np.array([np.interp(f, range(1, 21), self.LOOKUP_TABLE[i]) for i in range(self.N)])
    
    def __str__(self):
        print("Observation space: ", self.observation_space)
        print("Action space: ", self.action_space)
        return ""



from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import os
class LogWeightsCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self):
        # Initialize the writer for TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _on_step(self):
        # Log weights at every step
        step = self.num_timesteps
        model_params = self.model.policy.state_dict()
        for param_name, param_value in model_params.items():
            self.writer.add_histogram(f"weights/{param_name}", param_value, step)
        return True

    def _on_training_end(self):
        # Close the writer
        if self.writer is not None:
            self.writer.close()


# def run_one_episode():
#     env = uBotsGym(N=2)  #, render_mode="human")
#     print("Observation space: ", env.observation_space)
#     print("Action space: ", env.action_space)
#     obs, info = env.reset()
#     for i in range(100):
#         # action = env.action_space.sample()
#         action = (0.1, np.pi / 4)
#         obs, reward, terminated, truncated, info = env.step(action)
#         env.render()
#     env.close()


def make_single_env(env_kwargs):
    def _init():
        env = uBotsGym(N=2, **env_kwargs)
        return env
    return _init


def train(alg='ppo', env_kwargs=None, device = 'cpu'):
    '''RL training function'''

    # Create environment. Multiple parallel/vectorized environments for faster training
    env = make_vec_env(make_single_env(env_kwargs), n_envs=48)
    # env = make_single_env(env_kwargs)()


    if alg == 'ppo':
        # PPO: on-policy RL
        # policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,  device= device)

        model = PPO(
                    CustomActorCriticPolicy,  # Use the custom policy
                    env,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[256, 256], vf=[128, 256, 512, 256, 128]),      # Define network architecture
                        activation_fn=th.nn.ReLU  # Ensure hidden layers use ReLU
                    ),
                    verbose=1,
                    device=device
                )
    elif alg == 'DQN':
        # DQN: off-policy RL
        policy_kwargs = dict(net_arch=[64, 64, 64, 64])
        model = DQN("MlpPolicy", env, verbose=1,  device= device, policy_kwargs=policy_kwargs)

    else:
        # off-policy RL
        # policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        policy_kwargs = dict(net_arch=[64, 64, 64, 64])
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            # use_sde=True,
            # sde_sample_freq=8,
            learning_rate=0.001,
            learning_starts=5000,
            batch_size=2048,
            tau=0.05,
            gamma=0.99,
            # gradient_steps=1,
            verbose=1,
             device= device
        )

    # log the training params
    logfile = f"logs/{alg}_ubots"
    tb_logger = configure(logfile, ["stdout", "csv", "tensorboard"])
    model.set_logger(tb_logger)


    # Add the callback for logging weights
    weight_log_dir = os.path.join(logfile, "weights")
    # callback = LogWeightsCallback(log_dir=weight_log_dir)
    checkpoint_callback = CheckpointCallback(
                save_freq=1000,
                save_path = weight_log_dir,
                name_prefix="rl_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
                )
    eval_env = uBotsGym(N=2, render_mode="human", **env_kwargs)
    best_model_save_path = os.path.join(weight_log_dir, "best_model")
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path,
                             log_path=weight_log_dir, eval_freq=10000,
                             deterministic=True, render=False)
    callback = CallbackList([checkpoint_callback, eval_callback])


    # train the model
    model.learn(10e6, progress_bar=True, callback=callback)

    model.save(models_dir / f"{alg}_ubots")
    del model
    env.close()


def evaluate(alg, env_kwargs, n_trials=10):
    '''Evaluate the trained RL model'''

    # create single environment for evaluation
    env = uBotsGym(N=2, render_mode="human", **env_kwargs)
    if alg == 'ppo':
        ALG = PPO
    else:
        ALG = SAC
    
    # load trained RL model

    # model = ALG.load(models_dir / f"{alg}_ubots", env=env)
    best_model_dir = './logs/ppo_ubots/weights/best_model/best_model.zip'
    model = ALG.load(best_model_dir, env=env)

    # run some episodes (trials)
    for trial in range(n_trials):
        obs, info = env.reset()
      
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward}")
            print(f"obs: {obs}", 'shape:', obs.shape)
            print(f"action: {action}", 'shape:', action.shape)
            print("info: ", info)
            done = terminated or truncated
            env.render()
        print(f"Trial: {trial}, Success: {info['is_success']}, # Successes = {info['n_successes']}")
    env.close()




def test_custom_network(env_kwargs, n_trials=10, device='cpu', type ='cont'):
    '''Load a trained PyTorch model and test its performance in the environment'''

    # Load the environment
    env = uBotsGym(N=2, render_mode="human", **env_kwargs)

    # Load the trained neural network
    
    if type == "disc":

        model = MLP(input_size=4, num_discrete_actions=20).to(device) # Must match the original model architecture

        # Load the saved state dictionary
        model.load_state_dict(th.load("mlp_model.pth"))
        
        model.eval()  # Set to evaluation mode

        # Run multiple trials
        for trial in range(n_trials):
            obs, info = env.reset()
            obs = obs[0:2].flatten()
            obs = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)  # Add batch dimension

            done = False
            while not done:
                with th.no_grad():  # No gradients needed for inference
                    state = th.tensor(obs, dtype=th.float32).flatten().to(device).unsqueeze(0)
                    action,heading = model.infer_action(state)  # Predict action
                obs, reward, terminated, truncated, info = env.step(np.array([action, heading]))
                obs = obs[0:2].flatten()
                obs = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)  # Add batch dimension
                # obs = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)  # Update obs for next step
                done = terminated or truncated
                env.render()

            print(f"Trial {trial + 1}: Success = {info['is_success']}, # Successes = {info['n_successes']}")

    if type == "cont":
        model = MLP_continuous(input_size=4, output_size=2).to(device) # Must match the original model architecture

        # Load the saved state dictionary
        model.load_state_dict(th.load("mlp_model_continuous.pth"))
        
        model.eval()  # Set to evaluation mode
        for trial in range(n_trials):
            obs, info = env.reset()
            obs = obs[0:2].flatten()
            obs = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)  # Add batch dimension
            

            done = False
            while not done:
                with th.no_grad():  # No gradients needed for inference
                    state = th.tensor(obs, dtype=th.float32).flatten().to(device).unsqueeze(0)
                    action,heading = model(state)  # Predict action
                obs, reward, terminated, truncated, info = env.step([action.squeeze().numpy(), heading.squeeze().numpy()])
                obs = obs[0:2].flatten()
                obs = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)  # Add batch dimension
                # obs = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)  # Update obs for next step
                done = terminated or truncated
                env.render()

            print(f"Trial {trial + 1}: Success = {info['is_success']}, # Successes = {info['n_successes']}")

    env.close()


import torch.nn as nn
# class MLP(nn.Module):
#     def __init__(self, input_size=4, output_size=4):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 256)
#         self.fc4 = nn.Linear(256, 4)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.fc4(x)
#         return x



class MLP_continuous(nn.Module):
    def __init__(self, input_size=4, output_size=2):
        super(MLP_continuous, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
         
        # Output layers
        self.fc_action = nn.Linear(128, 1)  # Discrete action classification
        self.fc_heading = nn.Linear(128, 1)  # Continuous heading direction
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        action = self.relu(self.fc_action(x))
        heading = self.fc_heading(x)
        return action, heading

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_discrete_actions=20):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(2*hidden_size))
        self.fc3 = nn.Linear(int(2*hidden_size), hidden_size)
        self.relu = nn.ReLU()
        
        # Output layers
        self.fc_action = nn.Linear(hidden_size, num_discrete_actions)  # Discrete action classification
        self.fc_heading = nn.Linear(hidden_size, 1)  # Continuous heading direction

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        action_logits = self.fc_action(x)  # No activation (will apply softmax in loss function)
        heading = self.fc_heading(x)  # No activation (direct regression for heading)
        return action_logits, heading
    def infer_action(self, state):
            # Add batch dimension
        
        action_logits, heading = self.forward(state)
        action_logits = action_logits.flatten()
        action_probabilities = th.softmax(action_logits, dim = 0)
        action = th.argmax(action_probabilities).item()  # Get the discrete action
        heading = heading.item()  # Get the continuous heading value
        return action, heading




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",
                        action="store_true", 
                        default=False, 
                        help="Runs the evaluation of a trained model. Default: False (runs RL training by default)")
    args = parser.parse_args()
    args.eval = True

    # create directory for saving RL models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # create directory for logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # set environment params
    env_kwargs = dict(XMIN=-100,
                 XMAX=100,
                 YMIN=-100, 
                 YMAX=100,
                 horizon=120)

    # run_one_episode()
    alg = ['DQN','ppo', 'sac'][1]
    if not args.eval:
        # if training
        train(alg, env_kwargs)
    else:
        # if evaluating
        evaluate(alg, env_kwargs)
        test_custom_network( env_kwargs=env_kwargs, n_trials=1, device='cpu',  type ='disc')