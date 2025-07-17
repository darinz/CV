# Contrastive Learning in Self-Supervised Learning

Contrastive learning is a powerful approach in self-supervised learning that trains models to bring similar (positive) pairs closer and push dissimilar (negative) pairs apart in the embedding space. Below, we cover the core concepts, mathematical foundations, and popular frameworks, with detailed explanations and Python code examples.

---

## 1. Core Concepts

- **Positive Pair:** Two augmented views of the same instance (e.g., two crops of the same image).
- **Negative Pair:** Views from different instances (e.g., crops from different images).

The goal is to learn an embedding where positive pairs are close and negative pairs are far apart.

---

## 2. Contrastive Loss (InfoNCE)

### Explanation
- InfoNCE is a popular loss for contrastive learning.
- For a query $q$, a positive key $k^+$, and a set of negative keys $\{k^-_j\}$, the loss encourages $q$ to be close to $k^+$ and far from all $k^-_j$.

### Mathematical Formulation
$$
L = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_j \exp(q \cdot k^-_j / \tau)}
$$
where $\tau$ is a temperature hyperparameter.

### Python Example (PyTorch)
```python
import torch
import torch.nn.functional as F

def info_nce_loss(query, positive_key, negative_keys, temperature=0.07):
    # query: (D,), positive_key: (D,), negative_keys: (N, D)
    query = F.normalize(query, dim=0)
    positive_key = F.normalize(positive_key, dim=0)
    negative_keys = F.normalize(negative_keys, dim=1)
    pos_sim = torch.exp(torch.dot(query, positive_key) / temperature)
    neg_sim = torch.exp(torch.matmul(negative_keys, query) / temperature).sum()
    loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    return loss
```

---

## 3. SimCLR Framework

### Explanation
- SimCLR uses strong data augmentations to create two views of each image.
- Both views are encoded by a shared network, projected to a lower-dimensional space, and trained with contrastive loss.

### Steps
1. Apply random augmentations to an image to create two views $x_i, x_j$.
2. Encode with a shared network $f(\cdot)$ to get representations $h_i, h_j$.
3. Project to a lower-dimensional space $z_i, z_j$.
4. Use contrastive loss to maximize agreement between $z_i$ and $z_j$.

### Python Example (PyTorch, simplified)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRNet(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# Example usage:
# x1, x2: (batch, C, H, W) - two augmented views
# encoder: a CNN returning (batch, 512)
# model = SimCLRNet(encoder)
# z1 = model(x1)
# z2 = model(x2)
# Use info_nce_loss for each positive pair in the batch
```

---

## 4. MoCo (Momentum Contrast)

### Explanation
- MoCo maintains a dynamic dictionary (queue) of negative samples and uses a momentum encoder to update key representations.
- This allows for a large and consistent set of negatives.

### Key Ideas
- **Query Encoder:** Encodes the current batch.
- **Key Encoder:** Momentum-updated copy for the dictionary.
- **Queue:** Stores previous key representations as negatives.

### Python Example (PyTorch, simplified)
```python
import torch
import torch.nn as nn

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999):
        super().__init__()
        self.query_encoder = base_encoder
        self.key_encoder = base_encoder
        self.K = K
        self.m = m
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    # ... (enqueue/dequeue logic omitted for brevity)
```

---

## 5. BYOL (Bootstrap Your Own Latent)

### Explanation
- BYOL learns by predicting one view from another without using negative pairs.
- Uses an online and a target network, both with encoders and projectors.

### Key Ideas
- **Online Network:** Learns to predict the target network's output.
- **Target Network:** Slowly updated moving average of the online network.

### Python Example (PyTorch, simplified)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.online_encoder = base_encoder
        self.online_projector = nn.Linear(512, projection_dim)
        self.target_encoder = base_encoder
        self.target_projector = nn.Linear(512, projection_dim)
    @torch.no_grad()
    def update_target(self, m=0.996):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * m + param_o.data * (1. - m)
    def forward(self, x1, x2):
        # Online network
        z1_online = self.online_projector(self.online_encoder(x1))
        z2_online = self.online_projector(self.online_encoder(x2))
        # Target network
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))
        return z1_online, z2_online, z1_target, z2_target
# Loss: MSE between normalized online and target projections
```

---

Contrastive learning and its variants have enabled significant advances in self-supervised representation learning, especially in computer vision. 