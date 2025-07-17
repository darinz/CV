# Robotic Manipulation

Robotic manipulation focuses on controlling robot arms or hands to interact with objects. This guide covers grasping, imitation learning, visual servoing, and reinforcement learning for manipulation, with detailed explanations and Python code examples.

---

## 1. Grasping

Grasping involves predicting grasp points or poses from sensory input (e.g., images, depth, tactile).

**Grasp Quality Metric:**

$$
Q_{grasp} = f_{\text{grasp}}(s, o)
$$
where $s$ is the state and $o$ is the object.

**Python Example (Grasp Quality Prediction):**
```python
import torch
import torch.nn as nn

class GraspNet(nn.Module):
    def __init__(self, state_dim, obj_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + obj_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, state, obj):
        x = torch.cat([state, obj], dim=-1)
        return self.fc(x)

# Example usage:
state_dim = 4
obj_dim = 3
net = GraspNet(state_dim, obj_dim)
state = torch.randn(1, state_dim)
obj = torch.randn(1, obj_dim)
q_grasp = net(state, obj)
print("Grasp quality score:", q_grasp.item())
```

---

## 2. Imitation Learning

Imitation learning enables robots to learn manipulation skills from demonstrations.

**Behavioral Cloning:**

$$
\min_\theta \sum_{i} \| \pi_\theta(s_i) - a_i \|^2
$$

**Python Example:**
```python
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

policy_net = PolicyNet(state_dim, action_dim=2)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Dummy demonstration data
states = torch.randn(10, state_dim)
actions = torch.randn(10, 2)

# Behavioral cloning loss
pred_actions = policy_net(states)
loss = ((pred_actions - actions) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 3. Visual Servoing

Visual servoing uses visual feedback (e.g., camera images) to control the robot.

**Control Law:**

$$
a_t = f_{\text{servo}}(I_t, s_t)
$$
where $I_t$ is the image at time $t$.

**Python Example (Skeleton):**
```python
class VisualServoNet(nn.Module):
    def __init__(self, img_dim, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    def forward(self, img, state):
        x = torch.cat([img, state], dim=-1)
        return self.fc(x)

# Example usage:
img_dim = 8
state_dim = 4
action_dim = 2
net = VisualServoNet(img_dim, state_dim, action_dim)
img = torch.randn(1, img_dim)
state = torch.randn(1, state_dim)
action = net(img, state)
print("Servo action:", action)
```

---

## 4. Reinforcement Learning for Manipulation

RL can be used to learn end-to-end policies for tasks like pick-and-place, stacking, or tool use.

**Python Example (Skeleton):**
```python
# Assume use of a policy network as above
# Reward: +1 for successful manipulation, 0 otherwise

# Dummy data
state = torch.randn(1, state_dim)
action = policy_net(state)
reward = torch.tensor([1.0])  # Example reward

# Policy gradient update (as in Deep RL guide)
log_prob = torch.log(torch.abs(action).sum())  # Placeholder for real log-prob
loss = -log_prob * reward
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Summary

Robotic manipulation combines perception, control, and learning to enable robots to interact with the physical world. Approaches include grasping, imitation, visual feedback, and reinforcement learning. 