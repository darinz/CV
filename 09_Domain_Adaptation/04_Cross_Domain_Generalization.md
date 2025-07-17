# 4. Cross-Domain Generalization

Cross-domain generalization aims to train models that perform well on unseen domains, without explicit adaptation to each new domain. This guide covers the main concepts, mathematical formulations, and practical code examples.

---

## 4.1 Domain Generalization Problem

Given multiple training domains $\{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_M\}$, each with its own data distribution $P_i(x, y)$, the goal is to learn a model that generalizes to a new, unseen domain $\mathcal{D}_{M+1}$.

$$
P_i(x, y) \neq P_j(x, y), \quad \forall i \neq j
$$

---

## 4.2 Invariant Risk Minimization (IRM)

IRM aims to learn predictors that are invariant across different environments (domains).

**Objective:**
$$
\min_{f} \sum_{e \in \mathcal{E}} R^e(f)
$$
subject to:
$$
\arg\min_{\bar{f}} R^e(f \cdot \bar{f}) = \arg\min_{\bar{f}} R^{e'} (f \cdot \bar{f}), \quad \forall e, e' \in \mathcal{E}
$$

**IRM Penalty:**
$$
L_{IRM} = \sum_{e \in \mathcal{E}} R^e(f) + \lambda \sum_{e \in \mathcal{E}} \|\nabla_{w|w=1.0} R^e(w \cdot f)\|^2
$$

**Python Example (IRM Penalty):**
```python
import torch

def irm_penalty(loss, output, scale=1.0):
    grad = torch.autograd.grad(loss, output, create_graph=True)[0]
    return (grad * scale).pow(2).mean()
```

---

## 4.3 Group Distributionally Robust Optimization (Group DRO)

Group DRO minimizes the worst-case risk across predefined groups (domains).

**Objective:**
$$
\min_{f} \max_{q \in \Delta} \sum_{g=1}^{G} q_g R_g(f)
$$
where $\Delta$ is the probability simplex and $R_g(f)$ is the risk for group $g$.

**Python Example (Group DRO Loss):**
```python
import torch

def group_dro_loss(losses, group_ids, n_groups):
    group_losses = torch.stack([losses[group_ids == g].mean() for g in range(n_groups)])
    worst_group_loss = group_losses.max()
    return worst_group_loss
```

---

## 4.4 Mixup for Domain Generalization

Mixup is a data augmentation technique that creates new samples by interpolating between examples from different domains.

**Inter-domain Mixup:**
$$
x_{mix} = \lambda x_i + (1-\lambda) x_j
$$
$$
y_{mix} = \lambda y_i + (1-\lambda) y_j
$$
where $x_i, x_j$ come from different domains.

**Python Example:**
```python
import numpy as np

def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, y_mix
```

---

## Summary
- **IRM**: Learns invariant predictors across domains
- **Group DRO**: Minimizes worst-case group/domain risk
- **Mixup**: Augments data by mixing samples from different domains

These methods help models generalize to new, unseen domains. 