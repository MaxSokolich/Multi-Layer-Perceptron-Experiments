# RL-Gym setup for uBots Environment

## Installation

There is a `requirements.txt` file for the `pip` packages used to run the `ubots_gym.py` file.

### Setup

The `ubots_gym.py` implements a _Gymnasium_ class for the _uBots_ simulation, so it can be used with RL libraries like _Stable-Baselines3_.

Customizations:

1. Choose an on-policy RL (PPO) or off-policy (SAC)

2. Set the various environment parameters such as boundary coordinates, sampling time, etc.

3. `_get_goal()` function can generate random goals for each episode or set to fixed locations. Similarly, `__get_init_robot_pos()` can set a random or fixed initial locations for the robots.

4. `_get_reward()` creates a simple dense reward function based on Euclidean distance between robot and corresponding goal positions.

## Running experiments

To train the RL agent, use

```shell
$ python ubots_gym.py'
```

and evaluate trained agent with

```shell
$ python ubots_gym.py --eval'
```
