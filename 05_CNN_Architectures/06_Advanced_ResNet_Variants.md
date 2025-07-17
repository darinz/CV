# 06. Advanced ResNet Variants: ResNeXt and DenseNet

Modern CNN architectures build on the ideas of ResNet to further improve efficiency and accuracy. Two important variants are ResNeXt and DenseNet.

## ResNeXt

ResNeXt introduces the concept of **grouped convolutions** for better efficiency and representation power.

### Block Structure

A ResNeXt block aggregates outputs from multiple parallel transformations (groups):

```math
h^{(l+1)} = h^{(l)} + \sum_{g=1}^{G} \mathcal{F}_g(h^{(l)})
```

- $`G`$: Number of groups
- $`\mathcal{F}_g`$: Transformation for group $`g`$

### Benefits
- Increases model capacity without increasing computational cost
- Grouped convolutions are efficient on modern hardware

### Python Example: Grouped Convolution (PyTorch)

```python
import torch
import torch.nn as nn

# Grouped convolution: 32 groups
conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, groups=32)
x = torch.randn(8, 64, 32, 32)
y = conv(x)
print(y.shape)  # Output: torch.Size([8, 128, 32, 32])
```

## DenseNet

DenseNet connects each layer to all subsequent layers, promoting feature reuse and efficient gradient flow.

### Block Structure

Each layer receives as input the concatenation of all previous layers' outputs:

```math
h^{(l)} = \mathcal{H}_l([h^{(0)}, h^{(1)}, \ldots, h^{(l-1)}])
```

- $`[\cdot]`$: Concatenation
- $`\mathcal{H}_l`$: Transformation at layer $`l`$

### Benefits
- Improves gradient flow and feature reuse
- Reduces the number of parameters compared to traditional architectures

### Python Example: Dense Block (PyTorch)

```python
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1))
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)
```

## Summary
- **ResNeXt**: Uses grouped convolutions for efficient, powerful representations
- **DenseNet**: Connects all layers for maximum feature reuse and gradient flow
- Both architectures build on ResNet and are widely used in modern vision tasks

---

**Next:** [Training Strategies](07_Training_Strategies.md) 