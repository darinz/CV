# Model Learning in Robotics

Model learning involves learning a predictive model of the robot's environment or dynamics. This enables robots to plan and make decisions by simulating future outcomes. This guide covers the core concepts, mathematical foundations, and practical Python code examples.

---

## 1. Dynamics Model

A dynamics model predicts the next state $s'$ given the current state $s$ and action $a$:

$$
s' = f(s, a) \approx \hat{f}_\theta(s, a)
$$

- $f(s, a)$: True environment dynamics
- $\hat{f}_\theta(s, a)$: Learned model (e.g., neural network)

**Python Example:**
```python
import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

# Example usage:
state_dim = 4
action_dim = 2
model = DynamicsModel(state_dim, action_dim)
state = torch.randn(1, state_dim)
action = torch.randn(1, action_dim)
pred_next_state = model(state, action)
print("Predicted next state:", pred_next_state)
```

---

## 2. Model-Based Reinforcement Learning

Model-based RL uses the learned model $\hat{f}_\theta$ for planning or policy improvement.

### Model Predictive Control (MPC)
At each step, MPC solves:

$$
\max_{a_{0:H-1}} \sum_{t=0}^{H-1} R(s_t, a_t)
$$
subject to $s_{t+1} = \hat{f}_\theta(s_t, a_t)$

**Intuition:**
- Plan a sequence of actions that maximize expected reward over a horizon $H$.
- Only execute the first action, then re-plan at the next step.

**Python Example (Random Shooting MPC):**
```python
import numpy as np

def mpc_random_shooting(model, state, action_dim, horizon=5, num_candidates=100):
    best_return = -np.inf
    best_action = None
    for _ in range(num_candidates):
        actions = np.random.uniform(-1, 1, size=(horizon, action_dim))
        total_reward = 0
        s = state.copy()
        for a in actions:
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            s_next = model(s_tensor, a_tensor).detach().numpy().squeeze()
            reward = -np.linalg.norm(s_next)  # Example reward: keep state near zero
            total_reward += reward
            s = s_next
        if total_reward > best_return:
            best_return = total_reward
            best_action = actions[0]
    return best_action

# Example usage:
state = np.zeros(state_dim)
best_action = mpc_random_shooting(model, state, action_dim)
print("Best action from MPC:", best_action)
```

---

## 3. World Models

World models learn compact latent representations of the environment and use latent dynamics for planning and imagination-based RL.

- **Encoder:** Maps observations to latent space.
- **Latent Dynamics:** Predicts next latent state.
- **Decoder:** Maps latent state back to observation.

**Python Example (Skeleton):**
```python
class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, obs):
        return self.fc(obs)

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim)
        )
    def forward(self, z):
        return self.fc(z)

# Example usage:
obs_dim = 8
latent_dim = 4
encoder = Encoder(obs_dim, latent_dim)
dynamics = LatentDynamics(latent_dim, action_dim)
decoder = Decoder(latent_dim, obs_dim)

obs = torch.randn(1, obs_dim)
action = torch.randn(1, action_dim)
z = encoder(obs)
z_next = dynamics(z, action)
obs_recon = decoder(z_next)
print("Reconstructed observation:", obs_recon)
```

---

## Summary

Model learning enables robots to predict and plan in complex environments, improving sample efficiency and enabling advanced control strategies like MPC and imagination-based RL. 