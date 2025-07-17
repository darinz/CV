# 05. ResNet

ResNet (Residual Network), introduced in 2015, revolutionized deep learning by enabling the training of very deep networks using skip (residual) connections.

## Residual Block

The fundamental building block of ResNet is the residual block:

```math
h^{(l+1)} = \mathcal{F}(h^{(l)}) + h^{(l)}
```

- $`\mathcal{F}(h^{(l)})`$: Residual function (e.g., two or three conv layers)
- $`h^{(l)}`$: Input to the block

### Why Residual Connections?
- Allow gradients to flow directly through skip connections
- Help avoid vanishing/exploding gradients
- Enable training of networks with 50, 100, or more layers

## Bottleneck Block

For deeper networks, ResNet uses a bottleneck design:

```math
h^{(l+1)} = h^{(l)} + W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot h^{(l)}))
```

- $`W_1`$: 1×1 conv (reduces channels)
- $`W_2`$: 3×3 conv
- $`W_3`$: 1×1 conv (restores channels)

## Mathematical Properties

### Gradient Flow

The gradient can flow directly through skip connections:

```math
\frac{\partial L}{\partial h^{(l)}} = \frac{\partial L}{\partial h^{(l+1)}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial h^{(l)}}\right)
```

### Identity Mapping

If $`\mathcal{F}(h^{(l)}) = 0`$:
```math
h^{(l+1)} = h^{(l)}
```

This allows the network to learn identity mappings when optimal.

## ResNet Architectures

### ResNet-50 Example

- **Stage 1:** Conv(64, 7×7, stride=2) → MaxPool(3×3, stride=2)
  ```math
  \text{Output: } 56 \times 56 \times 64
  ```
- **Stage 2:** 3 bottleneck blocks, channels: 64→256
  ```math
  \text{Output: } 56 \times 56 \times 256
  ```
- **Stage 3:** 4 bottleneck blocks, channels: 128→512
  ```math
  \text{Output: } 28 \times 28 \times 512
  ```
- **Stage 4:** 6 bottleneck blocks, channels: 256→1024
  ```math
  \text{Output: } 14 \times 14 \times 1024
  ```
- **Stage 5:** 3 bottleneck blocks, channels: 512→2048
  ```math
  \text{Output: } 7 \times 7 \times 2048
  ```
- **Output:** GlobalAvgPool → FC(1000) → Softmax

### ResNet Variants
- **ResNet-18:** 18 layers, basic blocks
- **ResNet-34:** 34 layers, basic blocks
- **ResNet-50:** 50 layers, bottleneck blocks
- **ResNet-101:** 101 layers, bottleneck blocks
- **ResNet-152:** 152 layers, bottleneck blocks

## Python Example: Residual Block (PyTorch)

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
```

## Summary
- ResNet enables very deep networks by using residual connections
- Bottleneck blocks improve efficiency in deep variants
- ResNet is a foundation for many modern architectures

---

**Next:** [Advanced ResNet Variants](06_Advanced_ResNet_Variants.md) 