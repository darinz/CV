# 3. Meta-Learning for Computer Vision

Meta-learning, or "learning to learn," enables models to quickly adapt to new tasks with minimal data. This guide covers key meta-learning algorithms and their practical implementation.

---

## 3.1 Model-Agnostic Meta-Learning (MAML)

MAML learns a good initialization for model parameters so that the model can adapt quickly to new tasks with a few gradient steps.

### Mathematical Formulation
- **Inner Loop (Task-specific adaptation):**
  $$
  \theta_i' = \theta - \alpha \nabla_\theta L_{\mathcal{T}_i}(f_\theta)
  $$
- **Outer Loop (Meta-optimization):**
  $$
  \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} L_{\mathcal{T}_i}(f_{\theta_i'})
  $$

### Python Example (Simplified):
```python
import torch
import torch.nn as nn
import torch.optim as optim

def maml_step(model, loss_fn, x_spt, y_spt, x_qry, y_qry, alpha=0.01, beta=0.001):
    # Inner loop
    fast_weights = list(model.parameters())
    y_pred = model(x_spt)
    loss = loss_fn(y_pred, y_spt)
    grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
    fast_weights = [w - alpha * g for w, g in zip(fast_weights, grads)]
    # Outer loop
    y_pred_q = model(x_qry)
    meta_loss = loss_fn(y_pred_q, y_qry)
    meta_loss.backward()
    # (optimizer.step() outside)
```

---

## 3.2 Reptile

Reptile is a simpler meta-learning algorithm that moves the initialization towards the parameters found after task-specific training.

### Update Rule
$$
\theta \leftarrow \theta + \epsilon (\theta_i' - \theta)
$$
where $\epsilon$ is the meta-learning rate.

---

## 3.3 Prototypical Networks for Few-Shot

Prototypical Networks can be trained in an episode-based manner for meta-learning.

**Episode-based Training:**
1. Sample $N$ classes
2. Sample $K$ support examples per class
3. Sample $Q$ query examples per class

**Loss:**
$$
L = -\sum_{i=1}^{N \times Q} \log P(y_i = c_i|x_i, \mathcal{S})
$$

---

## 3.4 Relation Networks

Relation Networks learn a function to compare query and support examples.

**Relation Function:**
$$
r(x_q, x_i) = g([f_\theta(x_q), f_\theta(x_i)])
$$

**Classification:**
$$
P(y_q = c|x_q, \mathcal{S}) = \frac{\sum_{i: y_i = c} r(x_q, x_i)}{\sum_{i} r(x_q, x_i)}
$$

**Python Example (Relation Score):**
```python
import torch
import torch.nn as nn

class RelationNetwork(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(2 * feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x_q, x_s):
        # x_q: [B, D], x_s: [N, D]
        B, D = x_q.size()
        N = x_s.size(0)
        x_q_exp = x_q.unsqueeze(1).expand(B, N, D)
        x_s_exp = x_s.unsqueeze(0).expand(B, N, D)
        pair = torch.cat([x_q_exp, x_s_exp], dim=2)
        rel = self.g(pair).squeeze(-1)  # [B, N]
        return rel
```

---

## Summary
- **MAML**: Learns initialization for fast adaptation
- **Reptile**: Moves initialization towards post-adaptation weights
- **Prototypical/Relation Networks**: Meta-learned distance-based classifiers

Meta-learning is powerful for few-shot and fast adaptation scenarios in computer vision. 