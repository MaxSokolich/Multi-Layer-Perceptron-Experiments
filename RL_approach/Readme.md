# RL-Gym setup for uBots Environment

## Installation

There is a `requirements.txt` file for the `pip` packages used to run the `ubots_gym.py` file.

### Setup

The `ubots_env.py` implements a _Gymnasium_ class for the _uBots_ simulation, so it can be used with RL libraries like _Stable-Baselines3_. The base class is `uBotsGym` which implements the original continuous action-space; `uBotsGymDiscrete` inherits the base environment and discretizes the 2-D action-space (i.e., each action is `(freq, angle)`, discretized independently), and then enumerated as a Cartesian product.

Customizations:

1. Based on the nature of state-space of the Gym environment:
    - Continuous-space: Choose an on-policy RL (PPO) or off-policy (SAC, TD3) only. DQN does not work for continuous actions
    - Discrete-space: Choose models like DQN, PPO, A2C.

2. Set the various environment parameters such as boundary coordinates, sampling time, etc.

3. `_get_goal()` function can generate random goals for each episode or set to fixed locations. Similarly, `__get_init_robot_pos()` can set a random or fixed initial locations for the robots.

4. `_get_reward()` function defines custom rewards. See the function for vanilla geometric reward and then reward decomposition.

## Running experiments

NOTE: 

To train the RL agent, use

```shell
$ python main.py
```

and evaluate trained agent with

```shell
$ python main.py --eval
```
