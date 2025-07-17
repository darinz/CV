# Multimodal Fusion Strategies

Multimodal fusion strategies combine information from multiple modalities (e.g., text, image, audio) to create richer and more robust representations. This guide covers early fusion, late fusion, attention-based fusion, hierarchical fusion, and gated fusion, with math and code examples.

## 1. Introduction

Fusion strategies are essential for integrating diverse data sources in multi-modal learning. The choice of fusion method impacts model performance and interpretability.

## 2. Early Fusion

Combine modalities at the input level by concatenating raw or low-level features.

```math
x_{fusion} = \text{Concat}(x_1, x_2, \ldots, x_M)
```
```math
h = \text{Encoder}(x_{fusion})
```

#### Python Example: Early Fusion

```python
import numpy as np

def early_fusion(*modalities):
    return np.concatenate(modalities, axis=-1)
```

## 3. Late Fusion

Combine modalities after each is processed independently.

```math
h_i = \text{Encoder}_i(x_i)
```
```math
h_{fusion} = \text{Fusion}(h_1, h_2, \ldots, h_M)
```

#### Python Example: Late Fusion (Averaging)

```python
def late_fusion(*features):
    return sum(features) / len(features)
```

## 4. Attention-Based Fusion

Use attention mechanisms to selectively combine information from different modalities.

### 4.1 Cross-Modal Attention

```math
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k} \exp(e_{ik})}
```
```math
e_{ij} = \text{MLP}([h_i^{(1)}, h_j^{(2)}])
```
```math
h_i^{(1)} = h_i^{(1)} + \sum_{j} \alpha_{ij} h_j^{(2)}
```

#### Python Example: Cross-Modal Attention (Pseudocode)

```python
# Pseudocode for cross-modal attention
for i in range(len(h1)):
    e_ij = mlp(np.concatenate([h1[i], h2], axis=-1))
    alpha = softmax(e_ij)
    h1[i] = h1[i] + np.sum(alpha * h2, axis=0)
```

### 4.2 Multi-Head Cross-Attention

```math
\text{CrossAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

## 5. Hierarchical Fusion

Fuse information at multiple levels (local and global).

```math
h_{local} = \text{LocalFusion}(h_1, h_2)
```
```math
h_{global} = \text{GlobalFusion}(h_{local})
```

#### Python Example: Hierarchical Fusion (Pseudocode)

```python
# Pseudocode for hierarchical fusion
h_local = local_fusion(h1, h2)
h_global = global_fusion(h_local)
```

## 6. Gated Fusion

Learn a gate to control the contribution of each modality.

```math
g = \sigma(W_g [h_1, h_2] + b_g)
```
```math
h_{fusion} = g \odot h_1 + (1-g) \odot h_2
```

#### Python Example: Gated Fusion

```python
import torch
import torch.nn as nn

def gated_fusion(h1, h2):
    W_g = nn.Linear(h1.shape[-1] + h2.shape[-1], 1)
    b_g = nn.Parameter(torch.zeros(1))
    concat = torch.cat([h1, h2], dim=-1)
    g = torch.sigmoid(W_g(concat) + b_g)
    return g * h1 + (1 - g) * h2
```

## 7. Summary

Fusion strategies are crucial for effective multi-modal learning. Early, late, attention-based, hierarchical, and gated fusion each offer unique trade-offs in flexibility, interpretability, and performance. 