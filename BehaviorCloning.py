
from stable_baselines3.common.env_util import make_vec_env
import pickle
from imitation.algorithms.bc import BC
from ubots_env import *
from gym.wrappers import FlattenObservation
from stable_baselines3.common.logger import configure
# env = make_vec_env("YourEnvName-v0", n_envs=1)  # 
def make_single_env(env_kwargs):

    def _init():
       
        env = uBotsGymDiscrete(N=2, **env_kwargs)
        return env

    return _init
with open("trajectories_100k.pkl", "rb") as f:
    trajectories = pickle.load(f)


env_kwargs = dict(XMIN=-100,
                 XMAX=100,
                 YMIN=-100,
                 YMAX=100,
                 horizon=200
                )
    
env = make_vec_env(make_single_env(env_kwargs), n_envs=1)


from imitation.algorithms.bc import BC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
from torch import nn
import torch

# Set policy kwargs
policy_kwargs = dict(
    net_arch=[256, 128],
    activation_fn=nn.ReLU,
)



# Create policy manually
policy = ActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: 1e-3,
    net_arch=policy_kwargs["net_arch"],
    activation_fn=policy_kwargs["activation_fn"]
)



rng = np.random.default_rng(seed=0)  # 

# bc_trainer = BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=trajectories,
#     batch_size=int(4096*8),
#     ent_weight=0.0,
#     l2_weight=1e-4,  # NEW: directly use this
#     rng=rng  # required in imitation 1.0+
# )

bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectories,
    policy=policy,  # 👈 manually injected
    batch_size=int(4096*8),
    ent_weight=0.0,
    l2_weight=1e-4,  # NEW: directly use this
    rng=rng,  # required in imitation 1.0+
    device="cuda" if torch.cuda.is_available() else "cpu",
)

log_path = "./logs/bc_tensorboard/"
new_logger = configure(log_path, ["stdout", "tensorboard"])

bc_trainer.train(n_epochs=20) 
bc_trainer.policy.save("bc_policy_ubots100k.zip")


# Optional: Evaluate policy
# obs = env.reset()
# for _ in range(500):
#     action, _ = policy.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
