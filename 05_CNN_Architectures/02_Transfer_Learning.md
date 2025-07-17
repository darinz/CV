# 02. Transfer Learning

Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks, especially when labeled data is limited. It is widely used in computer vision.

## Pre-training Phase

A model is first trained on a large dataset (e.g., ImageNet):

```math
\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} L(f_\theta(x_i), y_i)
```

- $`\theta`$: Model parameters
- $`L`$: Loss function
- $`f_\theta`$: Model
- $`(x_i, y_i)`$: Training data

## Fine-tuning Strategies

### Feature Extraction
- **Freeze** pre-trained layers (do not update their weights)
- **Train** only new layers (e.g., classifier head)

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze all layers
model.fc = torch.nn.Linear(2048, 10)  # new head for 10 classes
```

### Fine-tuning
- **Update** all parameters, but use a lower learning rate for pre-trained layers

```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True
# Use a smaller learning rate for pre-trained layers
optimizer = torch.optim.SGD([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': 1e-4}
], momentum=0.9)
```

## Learning Rate Scheduling

### Differential Learning Rates

```math
\alpha_l = \begin{cases}
\alpha_{base} \cdot 0.1 & \text{for frozen layers} \\
\alpha_{base} & \text{for new layers}
\end{cases}
```

### Gradual Unfreezing

Progressively unfreeze layers during training:

```python
# Example: unfreeze one block at a time
for name, param in model.named_parameters():
    if 'layer4' in name:
        param.requires_grad = True
```

## Domain Adaptation

Transfer learning can be extended to domain adaptation, where the source and target data distributions differ.

### Maximum Mean Discrepancy (MMD)

```math
\text{MMD}(P, Q) = \left\|\mathbb{E}_{x \sim P}[\phi(x)] - \mathbb{E}_{y \sim Q}[\phi(y)]\right\|_{\mathcal{H}}^2
```

- $`P, Q`$: Source and target distributions
- $`\phi`$: Feature mapping
- $`\mathcal{H}`$: Reproducing kernel Hilbert space

## Summary
- Transfer learning uses pre-trained models to boost performance on new tasks
- Feature extraction and fine-tuning are common strategies
- Domain adaptation techniques help when data distributions differ

---

**Next:** [AlexNet](03_AlexNet.md) 