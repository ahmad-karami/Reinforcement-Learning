# REINFORCE & PPO algorithms

This notebook implements two on-policy deep reinforcement learning algorithms using PyTorch.


- The first part trains a REINFORCE agent on the classic control `CartPole-v1` environment, including environment exploration, policy network design, and return-based policy updates.

<p align="center">
<img src="https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/PPO/assets/CartPole.gif"  width="600">
</p
  
- The second part implements an actor-critic Proximal Policy Optimization (PPO) agent on the continuous-control `HalfCheetah` MuJoCo environment, covering policy and value networks, clipped surrogate objectives, and training loops for stable performance.


<p align="center">
<img src="https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/PPO/assets/Cheetah.gif"  width="600">
</p
