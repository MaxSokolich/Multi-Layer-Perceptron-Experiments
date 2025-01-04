from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Box

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env


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

        self.horizon = horizon
        self.continuous_task = continuous_task
        self.render_mode = render_mode

        # Set observation and action spaces
        self.observation_space = Box(
            low=np.array([[XMIN, YMIN], [XMIN, YMIN]]),  # Lower bounds for (x, y) of each robot
            high=np.array([[XMAX, YMAX], [XMAX, YMAX]]),  # Upper bounds for (x, y) of each robot
            shape=(N, 2),
            dtype=np.float32)
        self.action_space = Box(low=np.array([0, -np.pi]),
                                high=np.array([1, np.pi]))

        # Create matplotlib figure if rendering
        if render_mode == "human":
            self.fig, self.ax = plt.subplots()

    def reset(self, seed=None):
        # Set random seed
        self.observation_space.seed(seed)

        # Generate goal location at start of every episode
        self.goal0_pos, self.goal1_pos = self._get_goal()

        self._steps_elapsed = 0 # for checking horizon

        # create initial robot locations
        self.positions = self._get_init_robot_pos()
        
        obs = deepcopy(self.positions)

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

        obs = deepcopy(self.positions)

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
        rob0_pos = obs[0]
        rob1_pos = obs[1]

        # Calculate dist(robot, goal) for each robot
        d0 = np.linalg.norm(rob0_pos - self.goal0_pos)
        d1 = np.linalg.norm(rob1_pos - self.goal1_pos)

        # Check goal-reach condition
        successes = sum(np.array([d0, d1]) <= eps)

        # Calculate rewards
        # reward = -10.0 * (d0 + d1) + successes
        # reward = 10.0 * (np.exp(-d0) + np.exp(-d1))
        # reward = -1.0 * (np.exp(d0) + np.exp(d1))        
        reward = (1.0 - np.tanh(d0)) + (1.0 - np.tanh(d1))

        return reward, successes
    
    def _get_goal(self):
        # Random goal
        goal0, goal1 = np.random.uniform([5, 5], [self.XMAX, self.YMAX], (self.N, 2))
        
        # Fixed goal
        # goal0 = np.array([10, 10])
        # goal1 = np.array([10, 10])

        return goal0, goal1
    
    def _get_init_robot_pos(self):
        # Random goal
        rob0_pos, rob1_pos = np.random.uniform([self.XMIN, self.YMIN], [self.XMAX, self.YMAX], (self.N, 2))
        
        # Fixed positions
        # rob0_pos = np.array([0, 0])
        # rob1_pos = np.array([0, 0])

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


def run_one_episode():
    env = uBotsGym(N=2)  #, render_mode="human")
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    obs, info = env.reset()
    for i in range(100):
        # action = env.action_space.sample()
        action = (0.1, np.pi / 4)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()


def make_single_env(env_kwargs):
    def _init():
        env = uBotsGym(N=2, **env_kwargs)
        return env
    return _init


def train(alg='ppo', env_kwargs=None):
    '''RL training function'''

    # Create environment. Multiple parallel/vectorized environments for faster training
    env = make_vec_env(make_single_env(env_kwargs), n_envs=48)

    if alg == 'ppo':
        # PPO: on-policy RL
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

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
            learning_starts=1000,
            batch_size=2048,
            tau=0.05,
            gamma=0.95,
            # gradient_steps=1,
            verbose=1,
        )

    # log the training params
    logfile = f"logs/{alg}_ubots"
    tb_logger = configure(logfile, ["stdout", "csv", "tensorboard"])
    model.set_logger(tb_logger)

    # train the model
    model.learn(1_000_000, progress_bar=True)

    model.save(models_dir / f"{alg}_ubots")
    del model
    env.close()


def evaluate(alg, env_kwargs, n_trials=3):
    '''Evaluate the trained RL model'''

    # create single environment for evaluation
    env = uBotsGym(N=2, render_mode="human", **env_kwargs)
    if alg == 'ppo':
        ALG = PPO
    else:
        ALG = SAC
    
    # load trained RL model
    model = ALG.load(models_dir / f"{alg}_ubots", env=env)

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
    env_kwargs = dict(XMIN=-20,
                 XMAX=20,
                 YMIN=-20,
                 YMAX=20,
                 horizon=120)

    # run_one_episode()
    alg = ['ppo', 'sac'][1]
    if not args.eval:
        # if training
        train(alg, env_kwargs)
    else:
        # if evaluating
        evaluate(alg, env_kwargs)
