# 5. Domain-Invariant Representations

Learning domain-invariant representations is key to building models that generalize across different domains. This guide covers adversarial domain adaptation, MMD minimization, Wasserstein distance, and contrastive learning.

---

## 5.1 Adversarial Domain Adaptation

Adversarial training encourages the feature extractor to produce representations that are indistinguishable between source and target domains.

**Architecture:**
- **Feature Extractor** $G_f: \mathcal{X} \rightarrow \mathbb{R}^d$
- **Domain Discriminator** $G_d: \mathbb{R}^d \rightarrow [0, 1]$

**Objective:**
$$
\min_{G_f, G_y} \max_{G_d} L_y(G_y(G_f(x^s)), y^s) + \lambda L_d(G_d(G_f(x)), d)
$$

**Python Example (Domain Discriminator):**
```python
import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
```

---

## 5.2 Maximum Mean Discrepancy (MMD) Minimization

MMD can be used as a loss to align the distributions of source and target features.

**Loss:**
$$
L_{MMD} = \text{MMD}^2(f_\theta(\mathcal{X}_s), f_\theta(\mathcal{X}_t))
$$

**Python Example:**
(See Section 1.1 for code.)

---

## 5.3 Wasserstein Distance

Wasserstein distance measures the cost of transporting mass to transform one distribution into another. It is used for domain alignment.

**Definition:**
$$
W(\mu_s, \mu_t) = \inf_{\pi \in \Pi(\mu_s, \mu_t)} \mathbb{E}_{(x,y) \sim \pi}[\|x - y\|]
$$
where $\Pi(\mu_s, \mu_t)$ is the set of all couplings.

**Python Example (using POT library):**
```python
import numpy as np
import ot  # pip install POT

Xs = np.random.randn(100, 10)
Xt = np.random.randn(100, 10) + 1.0
M = ot.dist(Xs, Xt)
W_dist = ot.emd2([], [], M)
print(f"Wasserstein distance: {W_dist:.4f}")
```

---

## 5.4 Contrastive Learning for Domain Invariance

Contrastive learning encourages representations of similar (positive) pairs to be close and dissimilar (negative) pairs to be far apart.

**Loss:**
$$
L_{contrastive} = -\log \frac{\exp(sim(z_i, z_j^+)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(sim(z_i, z_k)/\tau)}
$$
where $sim$ is cosine similarity and $\tau$ is a temperature parameter.

**Python Example (NT-Xent Loss):**
```python
import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    N = z.size(0)
    mask = ~torch.eye(N, dtype=bool)
    sim = sim[mask].view(N, -1)
    positives = torch.cat([torch.diag(sim, N//2), torch.diag(sim, -N//2)])
    logits = sim / temperature
    labels = torch.arange(N)
    loss = F.cross_entropy(logits, labels)
    return loss
```

---

## Summary
- **Adversarial**: Domain discriminator encourages invariant features
- **MMD**: Aligns feature distributions
- **Wasserstein**: Measures optimal transport between domains
- **Contrastive**: Pulls together positive pairs, pushes apart negatives

These techniques are widely used for robust domain adaptation. 