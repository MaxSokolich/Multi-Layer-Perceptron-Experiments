from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Box

import gurobipy as gp
from gurobipy import GRB

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
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
    min_distance = 10**6
    
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
                0,2.72, 4.06, 5.80, 6.81, 9.07, 9.46, 11.32, 11.95, 14.11, 14.49,
                16.15, 16.49, 17.30, 17.09, 18.35, 19.68, 19.45, 21.39, 22.50,
                23.65
            ],
            [
                0,16.62, 27.30, 37.71, 48.13, 58.72, 66.67, 78.15,
                84.48, 96.43, 108.05, 119.22, 120.53, 127.00,
                133.90, 131.50, 151.17, 153.06, 161.49, 170.00,
                170.95
            ]]

    def __init__(self,
                 N, # number of uBots
                 XMIN=-10, # min x-coord
                 XMAX=10, # max x-coord
                 YMIN=-10, # min y-coord
                 YMAX=10, # max y-coord
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
        self.all_commands = np.load('all_commands_sparse.npy')
        # Set observation and action spaces
        # self.observation_space = Box(
        #     low=np.array([[XMIN, YMIN], [XMIN, YMIN]]),  # Lower bounds for (x, y) of each robot
        #     high=np.array([[XMAX, YMAX], [XMAX, YMAX]]),  # Upper bounds for (x, y) of each robot
        #     shape=(N, 2),
        #     dtype=np.float32)
        
        self.observation_space = gym.spaces.Dict(
            {
                "agents": Box(
                    low=np.array([[XMIN, YMIN], [XMIN, YMIN]]),  # Lower bounds for (x, y) of each robot
                    high=np.array([[XMAX, YMAX], [XMAX, YMAX]]),  # Upper bounds for (x, y) of each robot
                    shape=(N, 2),
                    dtype=np.float32),
                "goals": Box(
                    low=np.array([[XMIN, YMIN], [XMIN, YMIN]]),  # Lower bounds for (x, y) of each robot
                    high=np.array([[XMAX, YMAX], [XMAX, YMAX]]),  # Upper bounds for (x, y) of each robot
                    shape=(N, 2),
                    dtype=np.float32)
            }
        )
        self.frq_max =  5
        self.action_space = Box(low=np.array([0, -np.pi]),
                                high=np.array([self.frq_max, np.pi]))

        # Create matplotlib figure if rendering
        if render_mode == "human":
            self.fig, self.ax = plt.subplots()





    def check_in_bounds(self):
        #### check if the robots are within the bounds with some margin
        margin = 0.5
        out = np.all(self.positions[:, 0] > self.XMIN + margin) and np.all(self.positions[:, 0] < self.XMAX - margin) and np.all(self.positions[:, 1] > self.YMIN + margin) and np.all(self.positions[:, 1] < self.YMAX - margin)
        return out
    
    def reset(self, seed=None):
        # Set random seed
        self.observation_space.seed(seed)

        # Generate goal location at start of every episode
        self.goal0_pos, self.goal1_pos = self._get_goal()

        self._steps_elapsed = 0 # for checking horizon

        # create initial robot locations
        self.positions = self._get_init_robot_pos()
        
        # obs = deepcopy(self.positions)
        obs = {"agents": deepcopy(self.positions), "goals": self._get_goal()}

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
            positions = np.vstack(self.positions)
            self.scat = self.ax.scatter(positions[:, 0],
                                        positions[:, 1],
                                        c='b')

        return obs, info

    def step(self, action):
        f, alpha = action
        new_positions = []
        speeds = self.v_i(f)
        for i, pos in enumerate(self.positions):
            dx = speeds[i] * self.dt * np.cos(alpha)
            dy = speeds[i] * self.dt * np.sin(alpha)
            new_pos = pos + np.array([dx, dy])
            new_pos[0] = np.clip(new_pos[0], self.XMIN, self.XMAX)
            new_pos[1] = np.clip(new_pos[1], self.YMIN, self.YMAX)
            new_positions.append(new_pos)
        self.positions = np.array(new_positions)

        self._steps_elapsed += 1

        # obs = deepcopy(self.positions)
        obs = {"agents": deepcopy(self.positions), "goals": self._get_goal()}
        if not self.check_in_bounds():
            reward = -5000.0
            terminated = True
            truncated = True if (self._steps_elapsed >= self.horizon) else False
            info = {'is_success': False, 'n_successes': 0}
            successes = 0
        else:

            # Get reward and number of robots successfully reached their goals
            reward, successes = self._get_reward(obs)
      

        if self.continuous_task:
            terminated = False
        else:
            terminated = successes >= 2
        
        info = {'is_success': successes >= 2, 'n_successes': successes}
        
        truncated = True if (self._steps_elapsed >= self.horizon) else False

        return obs, reward, terminated, truncated, info

    def render(self):
        self.scat.set_offsets(self.positions)
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.1)

    def close(self):
        plt.close()

    def min_distance_to_boundary(self):
        min_distance = np.inf
        for pos in self.positions:
            min_distance = min(min_distance, pos[0] - self.XMIN, self.XMAX - pos[0], pos[1] - self.YMIN, self.YMAX - pos[1])
        return min_distance

    def _get_reward(self, obs, eps=1.0):
        """
        Calculate the rewards for current state.

        Parameters:
            obs: current observation
            eps: threshold for checking goal reach. Default: 1.0

        Returns:
            reward: the reward as a function of distance to goals
            successes: number of robots that successfully reached their corresponding goals            
        """
        # obs = obs["agents"]
        # rob0_pos = obs[0]
        # rob1_pos = obs[1]


   
        # d0 = np.linalg.norm(rob0_pos - self.goal0_pos)
        # d1 = np.linalg.norm(rob1_pos - self.goal1_pos)


      
        # successes = (np.array([d0, d1]) <= eps)
        # end_goal = successes.all()
        # end_goal_one_robot = successes.any()
        
        
        # minimum_distance_to_boundary = self.min_distance_to_boundary()
        # penalty_to_boundry = -20*np.exp(-0.1*minimum_distance_to_boundary)
        # reward = 80*end_goal -8*minimum_distance_to_path(self.expert_trj,np.array([rob0_pos, rob1_pos]).flatten())+penalty_to_boundry
        # print(f"Reward: {reward}")

        # return reward, successes


        
        obs = obs["agents"]
        rob0_pos = obs[0]
        rob1_pos = obs[1]
        state = np.array([rob0_pos, rob1_pos]).flatten()
        # Calculate dist(robot, goal) for each robot
        d0 = np.linalg.norm(rob0_pos - self.goal0_pos)
        d1 = np.linalg.norm(rob1_pos - self.goal1_pos)

        # Check goal-reach condition
        successes = np.all(np.array([d0, d1]) <= eps)

        # Calculate rewards
        end_goal = successes.all()
        # end_goal_one_robot = successes.any()
        
        
        minimum_distance_to_boundary = self.min_distance_to_boundary()
        penalty_to_boundry = -20*np.exp(-0.1*minimum_distance_to_boundary)
        # error_to_expert = minimum_distance_to_path(self.expert_trj,np.array([rob0_pos, rob1_pos]).flatten())

        # print(error_to_expert)
        # reward = -0.001 * (d0 + d1) + 150*successes+penalty_to_boundry

        cost_to_go = self.lp_reward(state)
        reward = -0.1*cost_to_go+150*successes
        print(f"cost_to_go: {cost_to_go}, reward: {reward}")

        # reward = 10.0 * (np.exp(-d0) + np.exp(-d1))
        # reward = -1.0 * (np.exp(d0) + np.exp(d1))        
        # reward = (1.0 - np.tanh(d0)) + (1.0 - np.tanh(d1))

        return reward, successes
    
    def _get_goal(self):
        # Random goal
        # goal0, goal1 = np.random.uniform([5, 5], [self.XMAX, self.YMAX], (self.N, 2))
        
        # Fixed goal
        goal0 = np.array([1.1, 12.1])
        goal1 = np.array([30.1, 35.1])

        return goal0, goal1
    
    def lp_reward(self, position):
        
        goal0, goal1 = self._get_goal()
        final_configuration = np.array([goal0, goal1]).flatten()
        size_T = len(self.all_commands)
        m = gp.Model()
        T = m.addMVar(size_T,ub = 100, lb = 0, name= 'Time periods')
        

        m.addConstr(position+self.all_commands.T@T == final_configuration)

        cost = gp.quicksum(np.linalg.norm(self.all_commands[i])*T[i] for i in range(size_T))
        cost += gp.quicksum(T)
       
        
        m.update()
        m.setObjective(cost, sense=gp.GRB.MINIMIZE)


        m.setParam('OutputFlag', 0)
        m.update()
        # m.params.NonConvex = 2
        m.optimize()
        if m.Status == gp.GRB.OPTIMAL:
           
            
            # print(f"Optimal solution found in {m.Runtime:.4f} seconds.")

            cost_to_go = 0
            for i in range(size_T):
                cost_to_go += np.linalg.norm(self.all_commands[i])*T[i].X
        else:
            print("No solution found.")
            cost_to_go = 10000
            

        return cost_to_go
    
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
        
        return np.array([np.interp(f, range(0, self.frq_max), self.LOOKUP_TABLE[i][:self.frq_max]) for i in range(self.N)])
    
    def __str__(self):
        print("Observation space: ", self.observation_space)
        print("Action space: ", self.action_space)
        return ""


def run_one_episode():
    env = uBotsGym(N=2)  #, render_mode="human")
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    obs, info = env.reset()
    for i in range(100):
        # action = env.action_space.sample()
        action = (0.1, np.pi / 4)
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()
    env.close()


def make_single_env(env_kwargs):
    def _init():
        env = uBotsGym(N=2, **env_kwargs)
        return env
    return _init


def train(alg='ppo', env_kwargs=None):
    '''RL training function'''

    # Create environment. Multiple parallel/vectorized environments for faster training
    env = make_vec_env(make_single_env(env_kwargs), n_envs=32)

    if alg == 'ppo':
        # PPO: on-policy RL
        # policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        policy_kwargs = dict(net_arch=[64, 64, 64, 64])
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    else:
        # off-policy RL
        # policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        policy_kwargs = dict(net_arch=[64, 64, 64, 64])
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            # use_sde=True,
            # sde_sample_freq=8,
            learning_rate=0.0003,
            learning_starts=1000,
            batch_size=512,
            tau=0.05,
            gamma=0.95,
            # gradient_steps=1,
            verbose=1,
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
    model.learn(1.5e7, progress_bar=True, callback=callback)

    model.save(models_dir / f"{alg}_ubots")
    del model

    # train the model
    # model.learn(1_000_000, progress_bar=True)

    

    # model.save(models_dir / f"{alg}_ubots")
    # del model
    # env.close()


def evaluate(alg, env_kwargs, n_trials=3):
    '''Evaluate the trained RL model'''

    # create single environment for evaluation
    env = uBotsGym(N=2, render_mode="human", **env_kwargs)
    if alg == 'ppo':
        ALG = PPO
    else:
        ALG = SAC
    
    # load trained RL model

    best_model_dir = './logs/sac_ubots/weights/best_model/best_model.zip'
    model = ALG.load(best_model_dir, env=env)
    # model = ALG.load(models_dir / f"{alg}_ubots", env=env)

    # run some episodes (trials)
    for trial in range(n_trials):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
        print(f"Trial: {trial}, Success: {info['is_success']}, # Successes = {info['n_successes']}")
    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",
                        action="store_true", 
                        default=False, 
                        help="Runs the evaluation of a trained model. Default: False (runs RL training by default)")
    args = parser.parse_args()

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
                 horizon=200)

    # run_one_episode(); exit()
    alg = ['ppo', 'sac'][1]
    # args.eval = True
    if not args.eval:
        # if training
        train(alg, env_kwargs)
    else:
        # if evaluating
        evaluate(alg, env_kwargs)
