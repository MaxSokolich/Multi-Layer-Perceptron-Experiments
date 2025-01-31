# networks/mlp_ppo.py

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor

class CustomMlpExtractor(MlpExtractor):
    """
    Custom MLP Extractor where the actor network uses ReLU activations for hidden layers
    and a Sigmoid activation for the output layer. The critic network uses ReLU activations
    throughout.
    """
    def _init_(self, feature_dim: int, net_arch: list, activation_fn: nn.Module = nn.ReLU):
        super(CustomMlpExtractor, self)._init_(feature_dim, net_arch, activation_fn=activation_fn)
        self._modify_actor_final_layer()

    def _modify_actor_final_layer(self):
        """
        Modifies the actor's final layer to include a sigmoid activation.
        """
        # Access the actor network (policy network)
        actor_net = self.policy_net

        # Reconstruct the actor network with a sigmoid activation at the end
        new_actor_layers = []
        for layer in actor_net:
            new_actor_layers.append(layer)
            if isinstance(layer, nn.Linear):
                # Assuming the last linear layer is the final layer before activation
                new_actor_layers.append(nn.Sigmoid())
        
        # Replace the policy_net with the new layers
        self.policy_net = nn.Sequential(*new_actor_layers)

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy for PPO with ReLU activations in hidden layers
    and a sigmoid activation in the actor's final layer.
    """
    def _init_(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        *args,
        **kwargs
    ):
        # Define default network architecture if not provided
        if net_arch is None:
            net_arch = [dict(pi=[256, 256], vf=[256, 256])]

        super(CustomActorCriticPolicy, self)._init_(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """
        Override the MLP extractor to use the custom extractor with modified actor network.
        """
        self.mlp_extractor = CustomMlpExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass through the network.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        values = self.value_net(latent_vf)
        return actions, values, log_prob

# from typing import Callable, Dict, List, Optional, Tuple, Type, Union
# import numpy as np

# from gymnasium import spaces
# import torch as th
# from torch import nn

# from stable_baselines3 import PPO
# from stable_baselines3.common.policies import ActorCriticPolicy


# class CustomNetwork(nn.Module):
#     """
#     Custom network for policy and value function.
#     It receives as input the features extracted by the features extractor.

#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         feature_dim: int = 8,
#         last_layer_dim_pi: int = 256,
#         last_layer_dim_vf: int = 256,
#     ):
#         super().__init__()

#         # IMPORTANT:
#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf

#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU(),
#             nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.ReLU(),
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(),
#             nn.Linear(last_layer_dim_vf, last_layer_dim_vf), nn.ReLU(),
#         )

#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         return self.policy_net(features)

#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         return self.value_net(features)


# class CustomActorCriticPolicy(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Callable[[float], float],
#         *args,
#         **kwargs,
#     ):
#         # Disable orthogonal initialization
#         kwargs["ortho_init"] = False
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )


#     def _build_mlp_extractor(self) -> None:
#         self.mlp_extractor = CustomNetwork(self.features_dim)

#     def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#         latent_pi, latent_vf = self.mlp_extractor(obs)
#         # Evaluate the values for the given observations
#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         # Use custom method for the action
#         actions = distribution.get_actions(deterministic=deterministic)
        
#         # Apply sigmoid to the last layer
#         actions = actions[0]
#         actions = th.sigmoid(actions)

#         actions[1] = actions[1] * 2 * np.pi - np.pi

#         # action[0] is between 0 and 1 but I need to map it to an index between 0 and 19
#         actions[0] = th.floor(actions[0] * 20)

#         return actions, values, distribution


