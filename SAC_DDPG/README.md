In **SAC_discrete.ipynb**, we implement and experiment with the Soft Actor-Critic (SAC) algorithm adapted for discrete action spaces using PyTorch on the classic `CartPole` environment.

### Algorithm Variants
- **Online SAC**: Standard model-free RL where the agent interacts with the environment during training, collecting fresh experiences via an experience replay buffer while maximizing both reward and entropy (exploration).
- **Offline SAC**: Trained purely on a fixed batch of pre-collected trajectories.
- **Behavioral Cloning (BC)**: Behavioral cloning works by mimicking expert behavior through supervised learning. It involves collecting a dataset of expert trajectories, which are sequences of state-action pairs (unlike offline reinforcement learning, which also collects rewards to train the model). The model is then trained to predict the expert's actions based on the observed states. The goal is for the model to replicate the expert's behavior by minimizing the difference between its predicted actions and the expert's actions.

This notebook compares their convergence, sample efficiency, and policy performance on `CartPole`'s balancing task.

<p align="center">
<img src="https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/SAC_DDPG/assets/CartPole.gif" alt="CartPole SAC vs DDPG" width="600">
</p>


In **SAC_DDPG_continuous.ipynb**, we implement and compare Deep Deterministic Policy Gradient (DDPG) and Soft Actor-Critic (SAC) algorithms for continuous action spaces using PyTorch in the HalfCheetah MuJoCo environment.

### Algorithm Implementations
- **Deep Deterministic Policy Gradient (DDPG)**: Off-policy actor-critic method using deterministic policies with separate actor and critic networks, employing target networks and noise injection for exploration in high-dimensional continuous control.
- **Soft Actor-Critic**
<p align="center">
<img src="https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/SAC_DDPG/assets/DDPG.png" alt="DDPG Architecture" width="600">
</p>

<p align="center">
<img src="https://github.com/ahmad-karami/Reinforcement-Learning/blob/main/SAC_DDPG/assets/SAC.png" alt="SAC Architecture" width="600">
</p>



