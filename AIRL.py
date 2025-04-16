import numpy as np
import torch
import os
import pickle

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
# from imitation.util.util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env
from imitation.util.networks import RunningNorm

from ubots_env import uBotsGym

# === Constants ===
SEED = 42
LOG_DIR = "./logs/airl_tensorboard/"
DEMO_PATH = "trajectories_100k_c.pkl"

# === Make environment ===
def make_env():
    return uBotsGym(N=2, XMIN=-100, XMAX=100, YMIN=-100, YMAX=100, horizon=200)

# venv = make_vec_env(make_env, n_envs=1, seed=SEED)


def make_single_env(env_kwargs):

    def _init():
       
        env = uBotsGym(N=2, **env_kwargs)
        return env

    return _init



env_kwargs = dict(XMIN=-100,
                 XMAX=100,
                 YMIN=-100,
                 YMAX=100,
                 horizon=200
                )
    
venv = make_vec_env(make_single_env(env_kwargs), n_envs=1)

# === Load expert demonstrations ===
with open(DEMO_PATH, "rb") as f:
    expert_trajectories = pickle.load(f)

# === Set up policy learner ===
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
)

# === Set up reward network ===
reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    use_done=True,
    reward_hid_sizes = (64,128,256),
    potential_hid_sizes = (64,128,128,256),
    normalize_input_layer=RunningNorm,  # ✅ correct normalization
)

# === Set up logger for TensorBoard ===
os.makedirs(LOG_DIR, exist_ok=True)
logger = configure(LOG_DIR, ["stdout", "tensorboard"])
_ = venv.reset()

learner.set_logger(logger) 
# === AIRL Trainer ===
airl_trainer = AIRL(
    demonstrations=expert_trajectories,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    # rng=np.random.default_rng(SEED),
    # logger=logger,
)

# === Evaluate before training ===
venv.seed(SEED)
rewards_before, _ = evaluate_policy(learner, venv, n_eval_episodes=10, return_episode_rewards=True)
print(f"🚀 Mean reward before AIRL training: {np.mean(rewards_before):.2f}")

# === Train AIRL ===
airl_trainer.train(total_timesteps=2_000_000)

# === Evaluate after training ===
venv.seed(SEED)
rewards_after, _ = evaluate_policy(learner, venv, n_eval_episodes=10, return_episode_rewards=True)
print(f"✅ Mean reward after AIRL training: {np.mean(rewards_after):.2f}")
print(f"🚀 Mean reward before AIRL training: {np.mean(rewards_before):.2f}")
# === Save final policy ===
learner.save("airl_policy_discrete.zip")
print("🎯 Trained AIRL policy saved as airl_policy_discrete.zip")

