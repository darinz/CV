# 2. Few-Shot and Zero-Shot Learning

Few-shot and zero-shot learning enable models to generalize to new classes with limited or no labeled examples. This guide covers the main concepts, mathematical formulations, and practical code examples.

---

## 2.1 Few-Shot Learning

### Problem Setup
Given:
- **Support set** $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{K \times N}$ (K examples for each of N classes)
- **Query set** $\mathcal{Q} = \{(x_i, y_i)\}_{i=1}^{Q}$

The model predicts:
$$
P(y_q|x_q, \mathcal{S}) = \sum_{c=1}^{C} P(y_q = c|x_q, \mathcal{S})
$$

### 2.1.1 Prototypical Networks
Prototypical Networks learn a prototype (mean embedding) for each class and classify queries by distance to prototypes.

**Prototype for class $c$:**
$$
\mu_c = \frac{1}{|\mathcal{S}_c|} \sum_{(x_i, y_i) \in \mathcal{S}_c} f_\theta(x_i)
$$

**Classification:**
$$
P(y_q = c|x_q, \mathcal{S}) = \frac{\exp(-d(f_\theta(x_q), \mu_c))}{\sum_{c'} \exp(-d(f_\theta(x_q), \mu_{c'}))}
$$
where $d$ is Euclidean distance.

**Python Example:**
```python
import torch
import torch.nn.functional as F

def euclidean_dist(a, b):
    # a: [N, D], b: [M, D]
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return ((a - b) ** 2).sum(2)

# Example: 5-way 1-shot
support = torch.randn(5, 128)  # 5 prototypes
query = torch.randn(10, 128)   # 10 queries
dists = euclidean_dist(query, support)
probs = F.softmax(-dists, dim=1)
print(probs)
```

### 2.1.2 Matching Networks
Matching Networks use an attention mechanism over the support set for classification.

**Classification:**
$$
P(y_q = c|x_q, \mathcal{S}) = \sum_{i=1}^{|\mathcal{S}|} a(x_q, x_i) \mathbb{1}[y_i = c]
$$

**Attention Function:**
$$
a(x_q, x_i) = \frac{\exp(c(f(x_q), g(x_i)))}{\sum_{j=1}^{|\mathcal{S}|} \exp(c(f(x_q), g(x_j)))}
$$
where $c$ is cosine similarity.

**Python Example:**
```python
import torch
import torch.nn.functional as F

def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a @ b.t())

support = torch.randn(5, 128)
query = torch.randn(10, 128)
sim = cosine_sim(query, support)
attn = F.softmax(sim, dim=1)
print(attn)
```

---

## 2.2 Zero-Shot Learning

### Problem Setup
- **Seen classes** $\mathcal{Y}_s$
- **Unseen classes** $\mathcal{Y}_u$
- $\mathcal{Y}_s \cap \mathcal{Y}_u = \emptyset$

### 2.2.1 Attribute-based Zero-Shot
Map both images and class attributes to a shared embedding space.

**Semantic Embedding:** $f_s: \mathcal{Y} \rightarrow \mathbb{R}^d$

**Visual Embedding:** $f_v: \mathcal{X} \rightarrow \mathbb{R}^d$

**Classification:**
$$
P(y|x) = \frac{\exp(f_v(x) \cdot f_s(y))}{\sum_{y' \in \mathcal{Y}_u} \exp(f_v(x) \cdot f_s(y'))}
$$

**Python Example:**
```python
import torch

# Assume visual features and semantic embeddings are normalized
x = torch.randn(1, 512)  # image feature
class_embeds = torch.randn(10, 512)  # 10 unseen classes
scores = (x @ class_embeds.t()).squeeze()
probs = torch.softmax(scores, dim=0)
print(probs)
```

### 2.2.2 Generalized Zero-Shot Learning
Handle both seen and unseen classes:
$$
P(y|x) = \frac{\exp(f_v(x) \cdot f_s(y) + \delta_y)}{\sum_{y' \in \mathcal{Y}_s \cup \mathcal{Y}_u} \exp(f_v(x) \cdot f_s(y') + \delta_{y'})}
$$
where $\delta_y$ is a calibration term.

---

## Summary
- **Few-Shot**: Learn from few examples (Prototypical, Matching Networks)
- **Zero-Shot**: Generalize to unseen classes using semantic embeddings

These methods are crucial for learning in low-data regimes and for open-world recognition. 