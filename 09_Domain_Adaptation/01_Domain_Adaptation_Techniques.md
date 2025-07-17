# 1. Domain Adaptation Techniques

Domain adaptation addresses the challenge of training a model on a source domain and adapting it to perform well on a target domain with a different data distribution.

## Problem Formulation

Suppose we have:
- Source domain: $\mathcal{D}_s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$
- Target domain: $\mathcal{D}_t = \{x_i^t\}_{i=1}^{n_t}$

The distributions differ:

$$
P_s(x, y) \neq P_t(x, y)
$$

The goal is to learn a model $f: \mathcal{X} \rightarrow \mathcal{Y}$ that performs well on the target domain.

---

## 1.1 Maximum Mean Discrepancy (MMD)

MMD is a statistical measure of the distance between two distributions. In domain adaptation, it is used to align the feature distributions of source and target domains.

**Mathematical Definition:**

$$
\text{MMD}^2(\mathcal{D}_s, \mathcal{D}_t) = \left\|\mathbb{E}_{x \sim P_s}[\phi(x)] - \mathbb{E}_{x \sim P_t}[\phi(x)]\right\|_{\mathcal{H}}^2
$$

where $\phi$ is a feature mapping to a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$.

**Empirical Estimate:**

$$
\text{MMD}^2 = \frac{1}{n_s^2} \sum_{i,j=1}^{n_s} k(x_i^s, x_j^s) + \frac{1}{n_t^2} \sum_{i,j=1}^{n_t} k(x_i^t, x_j^t) - \frac{2}{n_s n_t} \sum_{i=1}^{n_s} \sum_{j=1}^{n_t} k(x_i^s, x_j^t)
$$

where $k$ is a kernel function (e.g., Gaussian).

**Python Example:**
```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def compute_mmd(Xs, Xt, gamma=1.0):
    K_ss = rbf_kernel(Xs, Xs, gamma=gamma)
    K_tt = rbf_kernel(Xt, Xt, gamma=gamma)
    K_st = rbf_kernel(Xs, Xt, gamma=gamma)
    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd

# Example usage:
Xs = np.random.randn(100, 10)  # Source features
Xt = np.random.randn(100, 10) + 1.0  # Target features (shifted)
mmd_value = compute_mmd(Xs, Xt)
print(f"MMD^2: {mmd_value:.4f}")
```

---

## 1.2 Domain-Adversarial Neural Networks (DANN)

DANN uses adversarial training to learn features that are both discriminative for the main task and invariant to the domain.

**Architecture:**
- **Feature Extractor** $G_f: \mathcal{X} \rightarrow \mathbb{R}^d$
- **Label Predictor** $G_y: \mathbb{R}^d \rightarrow \mathcal{Y}$
- **Domain Discriminator** $G_d: \mathbb{R}^d \rightarrow \{0, 1\}$

**Training Objective:**

$$
L = L_y(G_y(G_f(x^s)), y^s) - \lambda L_d(G_d(G_f(x)), d)
$$

- $L_y$: classification loss (e.g., cross-entropy)
- $L_d$: domain classification loss
- $\lambda$: trade-off parameter

**Gradient Reversal Layer (GRL):**
During backpropagation, the GRL multiplies the gradient by $-\lambda$ for the domain loss, encouraging domain-invariant features.

**Python Example (PyTorch-like pseudocode):**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

class DANN(nn.Module):
    def __init__(self, feature_dim, class_num):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(784, feature_dim), nn.ReLU()
        )
        self.classifier = nn.Linear(feature_dim, class_num)
        self.domain = nn.Linear(feature_dim, 2)
    def forward(self, x, lambda_=1.0):
        feat = self.feature(x)
        class_pred = self.classifier(feat)
        domain_pred = self.domain(grad_reverse(feat, lambda_))
        return class_pred, domain_pred
```

---

## 1.3 Deep CORAL (Correlation Alignment)

Deep CORAL aligns the second-order statistics (covariances) of source and target features.

**Loss Function:**

$$
L_{CORAL} = \frac{1}{4d^2} \|C_s - C_t\|_F^2
$$

where $C_s$ and $C_t$ are covariance matrices of source and target features.

**Python Example:**
```python
import torch

def coral_loss(source, target):
    d = source.size(1)
    # Compute covariance
    def cov(m):
        m = m - m.mean(0, keepdim=True)
        return (m.t() @ m) / (m.size(0) - 1)
    Cs = cov(source)
    Ct = cov(target)
    loss = ((Cs - Ct) ** 2).sum() / (4 * d * d)
    return loss

# Example usage:
source = torch.randn(32, 128)  # batch of source features
 target = torch.randn(32, 128)  # batch of target features
loss = coral_loss(source, target)
print(f"CORAL loss: {loss.item():.4f}")
```

---

## Summary
- **MMD**: Aligns distributions by minimizing mean embedding distance
- **DANN**: Uses adversarial training for domain-invariant features
- **CORAL**: Aligns second-order statistics (covariances)

These techniques are foundational for domain adaptation in deep learning. 