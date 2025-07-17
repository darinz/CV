# Domain Adaptation and Transfer Learning

This module explores techniques for adapting models across different domains, enabling effective learning with limited data and improving generalization across diverse environments.

## Domain Adaptation Techniques

Domain adaptation addresses the problem of training a model on a source domain and adapting it to perform well on a target domain with different data distributions.

### Problem Formulation

Given source domain $\mathcal{D}_s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$ and target domain $\mathcal{D}_t = \{x_i^t\}_{i=1}^{n_t}$:

```math
P_s(x, y) \neq P_t(x, y)
```

The goal is to learn a model $f: \mathcal{X} \rightarrow \mathcal{Y}$ that performs well on the target domain.

### Maximum Mean Discrepancy (MMD)

MMD measures the distance between source and target distributions:

```math
\text{MMD}^2(\mathcal{D}_s, \mathcal{D}_t) = \left\|\mathbb{E}_{x \sim P_s}[\phi(x)] - \mathbb{E}_{x \sim P_t}[\phi(x)]\right\|_{\mathcal{H}}^2
```

where $\phi$ is a feature mapping to reproducing kernel Hilbert space $\mathcal{H}$.

#### Empirical MMD

```math
\text{MMD}^2 = \frac{1}{n_s^2} \sum_{i,j=1}^{n_s} k(x_i^s, x_j^s) + \frac{1}{n_t^2} \sum_{i,j=1}^{n_t} k(x_i^t, x_j^t) - \frac{2}{n_s n_t} \sum_{i=1}^{n_s} \sum_{j=1}^{n_t} k(x_i^s, x_j^t)
```

where $k$ is a kernel function.

### Domain-Adversarial Neural Networks (DANN)

DANN uses adversarial training to learn domain-invariant features.

#### Architecture

**Feature Extractor:**
```math
G_f: \mathcal{X} \rightarrow \mathbb{R}^d
```

**Label Predictor:**
```math
G_y: \mathbb{R}^d \rightarrow \mathcal{Y}
```

**Domain Discriminator:**
```math
G_d: \mathbb{R}^d \rightarrow \{0, 1\}
```

#### Training Objective

```math
L = L_y(G_y(G_f(x^s)), y^s) - \lambda L_d(G_d(G_f(x)), d)
```

where:
- $L_y$ is the classification loss
- $L_d$ is the domain classification loss
- $\lambda$ controls the trade-off between tasks

#### Gradient Reversal Layer

```math
\frac{\partial L_d}{\partial \theta_f} = -\lambda \frac{\partial L_d}{\partial G_f}
```

### Deep CORAL (Correlation Alignment)

CORAL aligns the second-order statistics of source and target features:

```math
L_{CORAL} = \frac{1}{4d^2} \|C_s - C_t\|_F^2
```

where $C_s$ and $C_t$ are the covariance matrices:

```math
C_s = \frac{1}{n_s-1} (X_s^T X_s - \frac{1}{n_s} (1^T X_s)^T (1^T X_s))
```

```math
C_t = \frac{1}{n_t-1} (X_t^T X_t - \frac{1}{n_t} (1^T X_t)^T (1^T X_t))
```

## Few-Shot and Zero-Shot Learning

Few-shot and zero-shot learning enable models to generalize to new classes with limited or no labeled examples.

### Few-Shot Learning

#### Problem Setup

Given support set $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{K \times N}$ and query set $\mathcal{Q} = \{(x_i, y_i)\}_{i=1}^{Q}$:

```math
P(y_q|x_q, \mathcal{S}) = \sum_{c=1}^{C} P(y_q = c|x_q, \mathcal{S})
```

#### Prototypical Networks

Learn prototypes for each class:

```math
\mu_c = \frac{1}{|\mathcal{S}_c|} \sum_{(x_i, y_i) \in \mathcal{S}_c} f_\theta(x_i)
```

**Distance-based Classification:**
```math
P(y_q = c|x_q, \mathcal{S}) = \frac{\exp(-d(f_\theta(x_q), \mu_c))}{\sum_{c'} \exp(-d(f_\theta(x_q), \mu_{c'}))}
```

where $d$ is Euclidean distance.

#### Matching Networks

Use attention mechanism over support set:

```math
P(y_q = c|x_q, \mathcal{S}) = \sum_{i=1}^{|\mathcal{S}|} a(x_q, x_i) \mathbb{1}[y_i = c]
```

**Attention Function:**
```math
a(x_q, x_i) = \frac{\exp(c(f(x_q), g(x_i)))}{\sum_{j=1}^{|\mathcal{S}|} \exp(c(f(x_q), g(x_j)))}
```

where $c$ is cosine similarity.

### Zero-Shot Learning

#### Problem Setup

Given seen classes $\mathcal{Y}_s$ and unseen classes $\mathcal{Y}_u$:

```math
\mathcal{Y}_s \cap \mathcal{Y}_u = \emptyset
```

#### Attribute-based Zero-Shot

**Semantic Embedding:**
```math
f_s: \mathcal{Y} \rightarrow \mathbb{R}^d
```

**Visual Embedding:**
```math
f_v: \mathcal{X} \rightarrow \mathbb{R}^d
```

**Classification:**
```math
P(y|x) = \frac{\exp(f_v(x) \cdot f_s(y))}{\sum_{y' \in \mathcal{Y}_u} \exp(f_v(x) \cdot f_s(y'))}
```

#### Generalized Zero-Shot Learning

Handle both seen and unseen classes:

```math
P(y|x) = \frac{\exp(f_v(x) \cdot f_s(y) + \delta_y)}{\sum_{y' \in \mathcal{Y}_s \cup \mathcal{Y}_u} \exp(f_v(x) \cdot f_s(y') + \delta_{y'})}
```

where $\delta_y$ is a calibration term.

## Meta-Learning for Computer Vision

Meta-learning (learning to learn) enables models to quickly adapt to new tasks.

### Model-Agnostic Meta-Learning (MAML)

MAML learns good initialization parameters for fast adaptation.

#### Inner Loop (Task-specific Adaptation)

```math
\theta_i' = \theta - \alpha \nabla_\theta L_{\mathcal{T}_i}(f_\theta)
```

#### Outer Loop (Meta-optimization)

```math
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} L_{\mathcal{T}_i}(f_{\theta_i'})
```

### Reptile

Simplified meta-learning algorithm:

```math
\theta \leftarrow \theta + \epsilon (\theta_i' - \theta)
```

where $\epsilon$ is the meta-learning rate.

### Prototypical Networks for Few-Shot

**Episode-based Training:**

For each episode:
1. Sample $N$ classes
2. Sample $K$ support examples per class
3. Sample $Q$ query examples per class

**Loss:**
```math
L = -\sum_{i=1}^{N \times Q} \log P(y_i = c_i|x_i, \mathcal{S})
```

### Relation Networks

Learn a relation function:

```math
r(x_q, x_i) = g([f_\theta(x_q), f_\theta(x_i)])
```

**Classification:**
```math
P(y_q = c|x_q, \mathcal{S}) = \frac{\sum_{i: y_i = c} r(x_q, x_i)}{\sum_{i} r(x_q, x_i)}
```

## Cross-Domain Generalization

Cross-domain generalization aims to improve performance across diverse domains without domain adaptation.

### Domain Generalization Problem

Given training domains $\{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_M\}$:

```math
P_i(x, y) \neq P_j(x, y), \quad \forall i \neq j
```

Goal: Learn a model that generalizes to unseen domain $\mathcal{D}_{M+1}$.

### Invariant Risk Minimization (IRM)

IRM learns invariant predictors across environments:

```math
\min_{f} \sum_{e \in \mathcal{E}} R^e(f)
```

subject to:
```math
\arg\min_{\bar{f}} R^e(f \cdot \bar{f}) = \arg\min_{\bar{f}} R^{e'} (f \cdot \bar{f}), \quad \forall e, e' \in \mathcal{E}
```

#### IRM Penalty

```math
L_{IRM} = \sum_{e \in \mathcal{E}} R^e(f) + \lambda \sum_{e \in \mathcal{E}} \|\nabla_{w|w=1.0} R^e(w \cdot f)\|^2
```

### Group Distributionally Robust Optimization (Group DRO)

Minimize worst-case risk across groups:

```math
\min_{f} \max_{q \in \Delta} \sum_{g=1}^{G} q_g R_g(f)
```

where $\Delta$ is the probability simplex and $R_g(f)$ is the risk for group $g$.

### Mixup for Domain Generalization

**Inter-domain Mixup:**
```math
x_{mix} = \lambda x_i + (1-\lambda) x_j
```

```math
y_{mix} = \lambda y_i + (1-\lambda) y_j
```

where $x_i, x_j$ come from different domains.

## Domain-Invariant Representations

Learning representations that are invariant across domains.

### Adversarial Domain Adaptation

#### Domain-Adversarial Training

**Feature Extractor:**
```math
G_f: \mathcal{X} \rightarrow \mathbb{R}^d
```

**Domain Discriminator:**
```math
G_d: \mathbb{R}^d \rightarrow [0, 1]
```

**Training Objective:**
```math
\min_{G_f, G_y} \max_{G_d} L_y(G_y(G_f(x^s)), y^s) + \lambda L_d(G_d(G_f(x)), d)
```

### Maximum Mean Discrepancy (MMD) Minimization

Minimize MMD between source and target features:

```math
L_{MMD} = \text{MMD}^2(f_\theta(\mathcal{X}_s), f_\theta(\mathcal{X}_t))
```

### Wasserstein Distance

Use Wasserstein distance for domain alignment:

```math
W(\mu_s, \mu_t) = \inf_{\pi \in \Pi(\mu_s, \mu_t)} \mathbb{E}_{(x,y) \sim \pi}[\|x - y\|]
```

where $\Pi(\mu_s, \mu_t)$ is the set of couplings.

### Contrastive Learning for Domain Invariance

**Positive Pairs:** Same class, different domains
**Negative Pairs:** Different classes

```math
L_{contrastive} = -\log \frac{\exp(sim(z_i, z_j^+)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(sim(z_i, z_k)/\tau)}
```

## Advanced Techniques

### Self-Training

Iteratively use model predictions as pseudo-labels:

```math
\mathcal{D}_t^{pseudo} = \{(x_i^t, \hat{y}_i^t) | \hat{y}_i^t = \arg\max_y P(y|x_i^t)\}
```

**Confidence Thresholding:**
```math
\mathcal{D}_t^{pseudo} = \{(x_i^t, \hat{y}_i^t) | \max_y P(y|x_i^t) > \tau\}
```

### Consistency Regularization

Enforce consistency under perturbations:

```math
L_{consistency} = \mathbb{E}_{x \sim \mathcal{D}_t} [\|f(x) - f(\text{Augment}(x))\|^2]
```

### Entropy Minimization

Minimize entropy on target domain:

```math
L_{entropy} = -\mathbb{E}_{x \sim \mathcal{D}_t} \left[\sum_{c=1}^{C} P(c|x) \log P(c|x)\right]
```

## Evaluation Metrics

### Domain Adaptation

#### Target Accuracy
```math
\text{Accuracy} = \frac{1}{|\mathcal{D}_t|} \sum_{(x,y) \in \mathcal{D}_t} \mathbb{1}[\arg\max_c P(c|x) = y]
```

#### H-score
```math
\text{H-score} = 2 \cdot \frac{\text{Accuracy} \cdot \text{Source Accuracy}}{\text{Accuracy} + \text{Source Accuracy}}
```

### Few-Shot Learning

#### N-way K-shot Accuracy
```math
\text{Accuracy} = \frac{1}{|\mathcal{Q}|} \sum_{(x,y) \in \mathcal{Q}} \mathbb{1}[\arg\max_c P(c|x, \mathcal{S}) = y]
```

### Zero-Shot Learning

#### Top-1 Accuracy
```math
\text{Top-1 Accuracy} = \frac{1}{|\mathcal{D}_u|} \sum_{(x,y) \in \mathcal{D}_u} \mathbb{1}[\arg\max_c P(c|x) = y]
```

#### Harmonic Mean
```math
\text{Harmonic Mean} = \frac{2 \cdot \text{Seen Accuracy} \cdot \text{Unseen Accuracy}}{\text{Seen Accuracy} + \text{Unseen Accuracy}}
```

## Applications

### Computer Vision

#### Image Classification
```math
f: \mathcal{X} \rightarrow \mathcal{Y}
```

#### Object Detection
```math
f: \mathcal{X} \rightarrow \{(b_i, c_i)\}_{i=1}^{N}
```

#### Semantic Segmentation
```math
f: \mathcal{X} \rightarrow \mathcal{Y}^{H \times W}
```

### Natural Language Processing

#### Text Classification
```math
f: \mathcal{T} \rightarrow \mathcal{Y}
```

#### Named Entity Recognition
```math
f: \mathcal{T} \rightarrow \mathcal{Y}^L
```

## Summary

Domain adaptation and transfer learning are essential for building robust AI systems:

1. **Domain Adaptation**: Techniques to adapt models across different data distributions
2. **Few-Shot Learning**: Learning new concepts with limited examples
3. **Zero-Shot Learning**: Generalizing to unseen classes
4. **Meta-Learning**: Learning to learn efficiently
5. **Cross-Domain Generalization**: Improving robustness across diverse domains
6. **Domain-Invariant Representations**: Learning features that generalize across domains

These techniques enable models to perform well in real-world scenarios with limited labeled data and diverse environments. 