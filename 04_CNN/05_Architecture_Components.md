# 05. CNN Architecture Components

Modern CNNs are built from a combination of key components. Understanding these building blocks is essential for designing and analyzing deep learning models.

## Convolutional Layer

Applies a set of learnable filters to the input:

```math
h^{(l)} = \sigma(W^{(l)} * h^{(l-1)} + b^{(l)})
```
- $`*`$ denotes convolution
- $`\sigma`$ is the activation function (e.g., ReLU)

## Pooling Layer

Reduces spatial dimensions:

```math
h^{(l)} = \text{Pool}(h^{(l-1)})
```

## Fully Connected Layer

Flattens the input and applies a linear transformation:

```math
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
```

## Activation Functions

- **ReLU:** $`\text{ReLU}(x) = \max(0, x)`$
- **Sigmoid:** $`\sigma(x) = \frac{1}{1 + e^{-x}}`$
- **Tanh:** $`\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}`$

## Python Example: Simple CNN Block

```python
import torch
import torch.nn as nn

class SimpleCNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)  # for 32x32 input
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## Summary
- CNNs are built from convolutional, pooling, and fully connected layers
- Activation functions introduce non-linearity
- These components are combined to form powerful deep learning models

---

**Next:** [Modern CNN Architectures](06_Modern_Architectures.md) 