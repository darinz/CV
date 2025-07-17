# 04. Pooling Operations in CNNs

Pooling operations reduce the spatial dimensions of feature maps while preserving important information. This helps control overfitting, reduces computation, and provides some translation invariance.

## Max Pooling

Selects the maximum value in each pooling region:

```math
\text{MaxPool}(I)_{i,j} = \max_{m,n \in \mathcal{R}_{i,j}} I_{m,n}
```

where $`\mathcal{R}_{i,j}`$ is the pooling region centered at $(i,j)$.

## Average Pooling

Computes the average value in each pooling region:

```math
\text{AvgPool}(I)_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{m,n \in \mathcal{R}_{i,j}} I_{m,n}
```

## Global Pooling

Reduces each feature map to a single value:

```math
\text{GlobalAvgPool}(I) = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} I_{i,j}
```
```math
\text{GlobalMaxPool}(I) = \max_{i,j} I_{i,j}
```

## Python Example: Pooling in PyTorch

```python
import torch
import torch.nn as nn

# Max pooling: 2x2 window, stride 2
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling: 2x2 window, stride 2
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# Example input: batch of 1 image, 1 channel, 4x4 pixels
x = torch.tensor([[[[1., 2., 3., 4.],
                   [5., 6., 7., 8.],
                   [9.,10.,11.,12.],
                   [13.,14.,15.,16.]]]])
print("MaxPool output:\n", maxpool(x))
print("AvgPool output:\n", avgpool(x))
```

## Summary
- Pooling reduces spatial size and computation
- Max pooling preserves strong activations; average pooling smooths features
- Global pooling is often used before fully connected layers or for classification

---

**Next:** [CNN Architecture Components](05_Architecture_Components.md) 