from pathlib import Path
import argparse

import numpy as np

from ubots_env import *

from stable_baselines3 import PPO, SAC, DQN, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


def run_one_episode():
    env = uBotsGymDiscreteHER(N=2)  #, render_mode="human")
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    
    # check_env(env)
    # raise

    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(i, action, obs)
        # env.render()
    env.close()


def make_single_env(env_kwargs):

    def _init():
        if ENV_TYPE == 'discrete':
            env = uBotsGymDiscreteHER(N=2, **env_kwargs)
        else:
            env = uBotsGym(N=2, **env_kwargs)
        return env

    return _init


def train(alg='ppo', env_kwargs=None):
    '''RL training function'''

    # Create environment. Multiple parallel/vectorized environments for faster training
    env = make_vec_env(make_single_env(env_kwargs), n_envs=16)

    if alg == 'ppo':
        # PPO: on-policy RL
        policy_kwargs = dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))
        # policy_kwargs = dict(net_arch=[64, 64, 64, 64])
        model = PPO("MultiInputPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    batch_size=256,
                    n_epochs=10,
                    n_steps=2048,
                    verbose=1)

    elif alg == 'sac':
        # SAC: off-policy RL
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
    else:
        # DQN: off-policy RL with Hindsight Experience Replay (HER)
        policy_kwargs = dict(net_arch=[512, 512, 512])
        model = DQN("MultiInputPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                        n_sampled_goal=4,
                        goal_selection_strategy="future",
                    ),
                    learning_starts=10_000,
                    batch_size=256,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    verbose=1)

    # log the training params
    logfile = f"logs/{alg}_ubots_{ENV_TYPE}"
    tb_logger = configure(logfile, ["stdout", "csv", "tensorboard"])
    model.set_logger(tb_logger)

    # train the model
    model.learn(10_000_000, progress_bar=True)

    model.save(models_dir / f"{alg}_ubots{ENV_TYPE}")
    del model
    env.close()


def evaluate(alg, env_kwargs, n_trials=3):
    '''Evaluate the trained RL model'''

    # create single environment for evaluation
    env = uBotsGymDiscreteHER(N=2, render_mode="human", **env_kwargs)
    if alg == 'ppo':
        ALG = PPO
    elif alg == 'sac':
        ALG = SAC
    else:
        ALG = DQN

    # load trained RL model
    model = ALG.load(models_dir / f"{alg}_ubots{ENV_TYPE}", env=env)

    # run some episodes (trials)
    for trial in range(n_trials):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
        print(
            f"Trial: {trial}, Success: {info['is_success']}, # Successes = {info['n_successes']}"
        )
    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help=
        "Runs the evaluation of a trained model. Default: False (runs RL training by default)"
    )
    args = parser.parse_args()

    # create directory for saving RL models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # create directory for logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    ENV_TYPE = 'discrete'  # 'continuous'

    # set environment params
    env_kwargs = dict(
        XMIN=-100,
        XMAX=100,
        YMIN=-100,
        YMAX=100,
        horizon=200,
    )

    # run_one_episode(); exit()
    alg = ['ppo', 'sac', 'dqn'][2]
    if not args.eval:
        # if training
        train(alg, env_kwargs)
    else:
        # if evaluating
        evaluate(alg, env_kwargs)
