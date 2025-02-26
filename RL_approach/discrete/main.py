from pathlib import Path
import argparse

import numpy as np

from ubots_env import *

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
import os

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
        if ENV_TYPE == 'discrete':
            env = uBotsGymDiscrete(N=2, **env_kwargs)
        else:
            env = uBotsGym(N=2, **env_kwargs)
        return env
    return _init


def train(alg='ppo', env_kwargs=None):
    '''RL training function'''

    # Create environment. Multiple parallel/vectorized environments for faster training
    env = make_vec_env(make_single_env(env_kwargs), n_envs=32)

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
    else:
        # PPO: on-policy RL
        # policy_kwargs = dict(net_arch=dict(pi=[512, 512], qf=[512, 512]))
        policy_kwargs = dict(net_arch=[512, 512]) 
        # policy_kwargs = dict(net_arch=[64, 64, 64, 64])
        model = DQN("MultiInputPolicy", 
                    env, 
                    policy_kwargs=policy_kwargs, 
                    learning_starts=400, 
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
    model.learn(10**6, progress_bar=True, callback=callback)

    model.save(models_dir / f"{alg}_ubots{ENV_TYPE}")
    del model
    env.close()


def evaluate(alg, env_kwargs, n_trials=3):
    '''Evaluate the trained RL model'''

    # create single environment for evaluation
    env = uBotsGymDiscrete(N=2, render_mode="human", **env_kwargs)
    if alg == 'ppo':
        ALG = PPO
    elif alg == 'sac':
        ALG = SAC
    else:
        ALG = DQN
    dir = '/home/mker/ubot_RL/Multi-Layer-Perceptron-Experiments/RL_approach/discrete/models/ppo_ubotsdiscrete_boundry.zip'
    # load trained RL model
    # model = ALG.load(models_dir / f"{alg}_ubots{ENV_TYPE}", env=env)
    model = ALG.load(dir, env=env)

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

    ENV_TYPE = 'discrete' # 'continuous'

    # set environment params
    env_kwargs = dict(XMIN=-100,
                 XMAX=100,
                 YMIN=-100,
                 YMAX=100,
                 horizon=200
                )
    
    # run_one_episode(); exit()
    alg = ['ppo', 'sac', 'dqn'][0]
    args.eval = True
    if not args.eval:
        # if training
        train(alg, env_kwargs)
    else:
        # if evaluating
        evaluate(alg, env_kwargs)
