# Deep Reinforcement Learning (Deep RL) for Robotics

Deep Reinforcement Learning (Deep RL) enables robots to learn control policies from high-dimensional sensory inputs (like images or joint angles) through trial and error. This guide covers the foundational concepts, mathematical formulations, and practical Python code examples.

---

## 1. Markov Decision Process (MDP)

A robot's environment is modeled as a Markov Decision Process (MDP):
- **State space** ($S$): All possible situations the robot can be in.
- **Action space** ($A$): All possible actions the robot can take.
- **Transition probability** ($P(s'|s,a)$): Probability of moving to state $s'$ from $s$ after action $a$.
- **Reward function** ($R(s,a)$): Immediate reward for taking action $a$ in state $s$.
- **Discount factor** ($\gamma$): How much future rewards are valued compared to immediate rewards.

**Mathematical Formulation:**

$$
(S, A, P, R, \gamma)
$$

**Python Example:**
```python
# Simple MDP environment using OpenAI Gym
import gym
env = gym.make('CartPole-v1')
state = env.reset()
for _ in range(10):
    action = env.action_space.sample()  # Random action
    next_state, reward, done, info = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
    state = next_state
    if done:
        break
```

---

## 2. Policy and Value Functions

- **Policy ($\pi(a|s)$):** Probability of taking action $a$ in state $s$.
- **State Value ($V^\pi(s)$):** Expected return starting from state $s$ following policy $\pi$.
- **Action Value ($Q^\pi(s, a)$):** Expected return starting from state $s$, taking action $a$, then following $\pi$.

**Math:**

$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s \right]
$$

$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s, a_0 = a \right]
$$

**Python Example:**
```python
def compute_return(rewards, gamma=0.99):
    """Compute discounted return for a list of rewards."""
    G = 0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return G

rewards = [1, 0, 2, 3]
print("Discounted return:", compute_return(rewards))
```

---

## 3. Deep Q-Learning

Deep Q-Learning uses a neural network to approximate the action-value function $Q(s, a)$. The agent learns to select actions that maximize expected rewards.

**Update Rule:**

$$
\theta \leftarrow \theta - \alpha \nabla_\theta (Q_\theta(s,a) - y)^2
$$
where $y = r + \gamma \max_{a'} Q_\theta(s', a')$

**Python Example:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

# Example usage:
state_dim = 4  # e.g., CartPole
action_dim = 2
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

# Dummy data
state = torch.randn(1, state_dim)
action = torch.tensor([1])
reward = torch.tensor([1.0])
next_state = torch.randn(1, state_dim)
done = False

gamma = 0.99
with torch.no_grad():
    target = reward + gamma * q_net(next_state).max(1)[0] * (1 - int(done))
q_value = q_net(state).gather(1, action.view(-1,1)).squeeze()
loss = (q_value - target).pow(2).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 4. Policy Gradient Methods

Policy gradient methods directly optimize the policy $\pi_\theta(a|s)$ by maximizing expected return.

**REINFORCE Gradient:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi}(s,a) ]
$$

**Python Example:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

policy_net = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Dummy data
state = torch.randn(1, state_dim)
action_probs = policy_net(state)
dist = torch.distributions.Categorical(action_probs)
action = dist.sample()
log_prob = dist.log_prob(action)
reward = torch.tensor([1.0])

# Policy gradient update
loss = -log_prob * reward  # Negative for gradient ascent
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 5. Actor-Critic Methods

Actor-Critic combines policy gradients (actor) with value function estimation (critic).
- **Actor:** Learns the policy $\pi_\theta(a|s)$
- **Critic:** Learns the value function $V_\phi(s)$ or $Q_\phi(s,a)$

**Python Example:**
```python
class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x)

critic_net = CriticNet(state_dim)

# Dummy data
value = critic_net(state)
advantage = reward - value  # Simplified advantage

# Actor loss (as above)
actor_loss = -log_prob * advantage.detach()
# Critic loss
critic_loss = advantage.pow(2).mean()

# Joint update
optimizer.zero_grad()
(actor_loss + critic_loss).backward()
optimizer.step()
```

---

## Summary

Deep RL provides a powerful framework for robots to learn complex behaviors from experience. By combining MDPs, value functions, and neural networks, robots can autonomously acquire skills for perception, planning, and control. 