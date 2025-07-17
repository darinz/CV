# 6. Advanced Techniques in Domain Adaptation

This guide covers advanced techniques that further improve domain adaptation: self-training, consistency regularization, and entropy minimization. Each method is explained with math and Python code examples.

---

## 6.1 Self-Training

Self-training uses model predictions as pseudo-labels for unlabeled target data, iteratively refining the model.

**Pseudo-labeling:**
$$
\mathcal{D}_t^{pseudo} = \{(x_i^t, \hat{y}_i^t) | \hat{y}_i^t = \arg\max_y P(y|x_i^t)\}
$$

**Confidence Thresholding:**
$$
\mathcal{D}_t^{pseudo} = \{(x_i^t, \hat{y}_i^t) | \max_y P(y|x_i^t) > \tau\}
$$

**Python Example:**
```python
import torch

def pseudo_labeling(model, x_t, threshold=0.9):
    with torch.no_grad():
        probs = torch.softmax(model(x_t), dim=1)
        conf, preds = probs.max(1)
        mask = conf > threshold
        return x_t[mask], preds[mask]
```

---

## 6.2 Consistency Regularization

Consistency regularization enforces that the model's predictions are consistent under input perturbations (e.g., augmentations).

**Loss:**
$$
L_{consistency} = \mathbb{E}_{x \sim \mathcal{D}_t} [\|f(x) - f(\text{Augment}(x))\|^2]
$$

**Python Example:**
```python
import torch

def consistency_loss(model, x):
    aug_x = x + 0.1 * torch.randn_like(x)  # simple noise augmentation
    out1 = model(x)
    out2 = model(aug_x)
    return ((out1 - out2) ** 2).mean()
```

---

## 6.3 Entropy Minimization

Entropy minimization encourages the model to make confident predictions on target data.

**Loss:**
$$
L_{entropy} = -\mathbb{E}_{x \sim \mathcal{D}_t} \left[\sum_{c=1}^{C} P(c|x) \log P(c|x)\right]
$$

**Python Example:**
```python
import torch
import torch.nn.functional as F

def entropy_minimization(model, x):
    probs = torch.softmax(model(x), dim=1)
    entropy = - (probs * torch.log(probs + 1e-8)).sum(1).mean()
    return entropy
```

---

## Summary
- **Self-Training**: Uses pseudo-labels for unlabeled data
- **Consistency Regularization**: Enforces prediction stability under perturbations
- **Entropy Minimization**: Encourages confident predictions

These advanced techniques are often combined with other domain adaptation methods for improved performance. 