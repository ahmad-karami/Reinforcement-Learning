# REINFORCE & PPO

This notebook implements two on-policy deep reinforcement learning algorithms using PyTorch.
- The first part trains a REINFORCE agent on the classic control `CartPole-v1` environment, including environment exploration, policy network design, and return-based policy updates.

![](https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/PPO/CartPole.gif)


- The second part implements an actor-critic Proximal Policy Optimization (PPO) agent on the continuous-control `HalfCheetah` MuJoCo environment, covering policy and value networks, clipped surrogate objectives, and training loops for stable performance.

![](https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/PPO/Cheetah.gif)
