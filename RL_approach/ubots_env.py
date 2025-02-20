import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

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
        self.action_space = Box(low=np.array([0, -np.pi]),
                                high=np.array([24, np.pi]))

        # Create matplotlib figure if rendering
        if render_mode == "human":
            self.fig, self.ax = plt.subplots()

    def reset(self, seed=None):
        # Set random seed
        self.observation_space.seed(seed)

        # Generate goal location at start of every episode
        self.goal0_pos, self.goal1_pos = self._get_goal()

        self._steps_elapsed = 0 # for checking horizon

        self.rob0_togo_prev = None
        self.rob1_togo_prev = None

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
        obs = obs["agents"]
        rob0_pos = obs[0]
        rob1_pos = obs[1]

        # Calculate dist(robot, goal) for each robot
        d0 = np.linalg.norm(rob0_pos - self.goal0_pos)
        d1 = np.linalg.norm(rob1_pos - self.goal1_pos)

        # Check goal-reach condition
        successes = sum(np.array([d0, d1]) <= eps)

        ### Option 1: Single reward function
        # reward = -10.0 * (d0 + d1) + successes
        # reward = 10.0 * (np.exp(-d0) + np.exp(-d1))
        # reward = -1.0 * (np.exp(d0) + np.exp(d1))        
        # reward = (1.0 - np.tanh(d0)) + (1.0 - np.tanh(d1))

        ### Option 2: Reward function decomposition (rob0 = Robot 0, rob1 = Robot 1)
        # Distance to goal
        rob0_dist = d0
        rob1_dist = d1

        # Size of step towards goal (energy cost)
        if self.rob0_togo_prev is None:
            rob0_progress = 0
        else:
            rob0_progress = (self.rob0_togo_prev - d0)
            self.rob0_togo_prev = d0

        if self.rob1_togo_prev is None:
            rob1_progress = 0
        else:
            rob1_progress = (self.rob1_togo_prev - d1)
            self.rob1_togo_prev = d1

        # Check if the robots overshoot their goals
        if (self.rob0_togo_prev or self.rob1_togo_prev) is None:
            all_overshoots = 0
        else:
            rob0_goal_overshoot = 1 if d0 > self.rob0_togo_prev else 0
            rob1_goal_overshoot = 1 if d0 > self.rob0_togo_prev else 0
            all_overshoots = rob0_goal_overshoot + rob1_goal_overshoot
        
        # Robots should reach their goals at approximately same times
        synchronous_reaching = abs(d0 - d1)

        # Compose the final reward
        alpha = 1
        beta = 1
        gamma = 1
        delta = 1
        lambda_ = 5
        mu = 1
        reward = (-alpha * rob0_dist) + \
                 (-beta * rob1_dist) + \
                 (gamma * rob0_progress) + \
                 (delta * rob1_progress) + \
                 (-lambda_ * all_overshoots) + \
                 (-mu * synchronous_reaching)

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
    

class uBotsGymDiscrete(uBotsGym):
    def __init__(self, N, XMIN=-10, XMAX=10, YMIN=-10, YMAX=10, dt=0.1, horizon=100, continuous_task=True, render_mode=None):
        super().__init__(N, XMIN, XMAX, YMIN, YMAX, dt, horizon, continuous_task, render_mode)

        n_actions = len(self.LOOKUP_TABLE[0]) # got from the number of entries in the LOOKUP_TABLE
        n_angles = 5 # feel free to change this number for better resolution
        self.dt_alpha = 2 * np.pi / n_angles

        self.action_space_cartesian_product = [(i, j) for i in range(n_actions) for j in range(n_angles)]

        # self.action_space = MultiDiscrete([n_actions, n_actions])
        self.action_space = Discrete(len(self.action_space_cartesian_product))

    def step(self, action):
        f_disc, alpha_disc = self.action_space_cartesian_product[action]
        new_positions = []

        f = self.LOOKUP_TABLE[0][f_disc]
        alpha = -np.pi + self.dt_alpha * alpha_disc
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

        # Get reward and number of robots successfully reached their goals
        reward, successes = self._get_reward(obs)

        if self.continuous_task:
            terminated = False
        else:
            terminated = successes >= 2
        
        info = {'is_success': successes >= 2, 'n_successes': successes}
        
        truncated = True if (self._steps_elapsed >= self.horizon) else False

        return obs, reward, terminated, truncated, info