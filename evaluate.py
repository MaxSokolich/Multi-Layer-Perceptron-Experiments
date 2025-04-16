import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import BasePolicy
from imitation.policies.serialize import load_policy
from ubots_env import uBotsGymDiscrete  # or uBotsGymDiscrete

# 1. Create environment
N = 2  # number of bots
rng = np.random.default_rng(0)

def make_single_env(env_kwargs):
    def _init():
        env = uBotsGymDiscrete(N=2, **env_kwargs)
        env = RolloutInfoWrapper(env)  # ✅ required for rollout()
        return env
    return _init
# with open("transitions.pkl", "rb") as f:
#     transitions = pickle.load(f)


env_kwargs = dict(XMIN=-100,
                 XMAX=100,
                 YMIN=-100,
                 YMAX=100,
                 horizon=200
                )

env = uBotsGymDiscrete(N=2, render_mode="human", **env_kwargs)

# 2. Load the policy
import warnings
from stable_baselines3.common.policies import ActorCriticPolicy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    policy = ActorCriticPolicy.load("bc_policy_ubots.zip")


def visualize_policy(policy, env, n_episodes=3):
    for trial in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                # print(obs)
                done = terminated or truncated
                env.render()
            print(f"Trial: {trial}, Success: {info['is_success']}, # Successes = {info['n_successes']}")
    env.close()


# Visualize
visualize_policy(policy, env)