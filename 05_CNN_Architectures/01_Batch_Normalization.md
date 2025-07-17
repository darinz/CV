# 01. Batch Normalization (BatchNorm)

Batch Normalization (BatchNorm) is a technique that normalizes the inputs of each layer, significantly improving training speed and stability of deep neural networks.

## Mathematical Formulation

For a mini-batch $`\mathcal{B} = \{x_1, x_2, \ldots, x_m\}`$:

### Training Phase

**Compute mini-batch statistics:**
```math
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i
```

```math
\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2
```

**Normalize:**
```math
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
```

**Scale and shift:**
```math
y_i = \gamma \hat{x}_i + \beta
```
where $`\gamma`$ and $`\beta`$ are learnable parameters, and $`\epsilon`$ is a small constant for numerical stability.

### Inference Phase

During inference, use running averages:
```math
\mu_{running} = \alpha \mu_{running} + (1 - \alpha) \mu_\mathcal{B}
```
```math
\sigma_{running}^2 = \alpha \sigma_{running}^2 + (1 - \alpha) \sigma_\mathcal{B}^2
```
```math
y_i = \gamma \frac{x_i - \mu_{running}}{\sqrt{\sigma_{running}^2 + \epsilon}} + \beta
```

## Benefits

1. **Faster Training**: Allows higher learning rates
2. **Reduced Internal Covariate Shift**: Stabilizes layer inputs
3. **Regularization Effect**: Adds noise during training
4. **Reduced Dependence on Initialization**: More robust to weight initialization

## Gradient Computation

The gradients with respect to the parameters are:
```math
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \hat{x}_i
```
```math
\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}
```
```math
\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \left(\frac{\partial L}{\partial y_i} - \frac{1}{m} \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} - \frac{\hat{x}_i}{m} \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \hat{x}_j\right)
```

## Variants

### Layer Normalization
Normalize across features for each sample:
```math
\mu_i = \frac{1}{H} \sum_{j=1}^{H} x_{i,j}
```
```math
\sigma_i^2 = \frac{1}{H} \sum_{j=1}^{H} (x_{i,j} - \mu_i)^2
```

### Instance Normalization
Normalize across spatial dimensions for each sample and channel:
```math
\mu_{i,c} = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{i,h,w,c}
```
```math
\sigma_{i,c}^2 = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{i,h,w,c} - \mu_{i,c})^2
```

## Python Example: BatchNorm Layer (PyTorch)

```python
import torch
import torch.nn as nn

# BatchNorm for 2D feature maps (e.g., images)
batchnorm = nn.BatchNorm2d(num_features=64)

# Example input: batch of 8 images, 64 channels, 32x32 pixels
x = torch.randn(8, 64, 32, 32)
y = batchnorm(x)
print(y.shape)  # Output: torch.Size([8, 64, 32, 32])
```

## Summary
- BatchNorm normalizes activations, improving training speed and stability
- Variants like LayerNorm and InstanceNorm are used in different contexts
- Widely used in modern deep learning architectures

---

**Next:** [Transfer Learning](02_Transfer_Learning.md) 