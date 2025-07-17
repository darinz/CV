# 3D Convolutional Neural Networks (3D CNNs)

3D CNNs extend 2D CNNs to process spatial-temporal data, making them well-suited for video analysis. This guide covers the basics of 3D convolution, the C3D architecture, and the I3D (Inflated 3D ConvNet) model, with detailed explanations and Python code examples.

---

## 1. 3D Convolution

### What is 3D Convolution?

A 3D convolution operates on a video clip (sequence of frames) instead of a single image. It captures both spatial (height, width) and temporal (time) information.

**Mathematical Formulation:**

$$
y_{i,j,k} = \sum_{c=1}^{C} \sum_{t=0}^{T-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{c,i+t,j+h,k+w} \cdot w_{c,t,h,w}
$$

- $x \in \mathbb{R}^{C \times T \times H \times W}$: input video clip
- $w \in \mathbb{R}^{C \times T \times H \times W}$: 3D kernel
- $y \in \mathbb{R}^{T' \times H' \times W'}$: output

### Python Example: 3D Convolution Layer
```python
import torch
import torch.nn as nn

# 3D convolution: in_channels, out_channels, kernel_size (T, H, W)
conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)

# Example input: batch of 8 videos, 3 channels, 16 frames, 112x112
x = torch.randn(8, 3, 16, 112, 112)
output = conv3d(x)
print(output.shape)  # (8, 64, 16, 112, 112)
```

---

## 2. C3D Architecture

C3D is a classic 3D CNN for video classification. It uses 3D convolutions and pooling to extract spatiotemporal features.

### C3D Block

$$
\text{C3D}(x) = \text{ReLU}(\text{BatchNorm}(\text{Conv3D}(x)))
$$

### C3D Network Structure

- **Input:** $V \in \mathbb{R}^{3 \times 16 \times 112 \times 112}$
- **Layer 1:** Conv3D(64, 3×3×3) → ReLU → MaxPool3D(1×2×2)
- **Layer 2:** Conv3D(128, 3×3×3) → ReLU → MaxPool3D(2×2×2)
- **Layer 3:** Conv3D(256, 3×3×3) → ReLU → MaxPool3D(2×2×2)
- **Layer 4:** Conv3D(512, 3×3×3) → ReLU → MaxPool3D(2×2×2)
- **Layer 5:** Conv3D(512, 3×3×3) → ReLU → MaxPool3D(2×2×2)

### Python Example: C3D Block
```python
class C3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Example usage:
# block = C3DBlock(3, 64)
# out = block(x)
```

---

## 3. I3D (Inflated 3D ConvNet)

I3D inflates 2D filters to 3D by repeating them across the temporal dimension, allowing the use of pretrained 2D weights.

### Filter Inflation

$$
W_{3D} = W_{2D} \otimes \mathbf{1}_T
$$

- $\otimes$: outer product
- $\mathbf{1}_T$: vector of ones (temporal dimension)

### I3D Inception Module

The I3D model uses 3D versions of Inception modules.

**Python Example: Inflating 2D Weights to 3D**
```python
import numpy as np

def inflate_2d_to_3d(weight_2d, time_dim):
    # weight_2d: (out_c, in_c, h, w)
    # Returns: (out_c, in_c, t, h, w)
    weight_3d = np.expand_dims(weight_2d, axis=2)
    weight_3d = np.repeat(weight_3d, time_dim, axis=2) / time_dim
    return weight_3d
```

---

## Summary

- 3D convolutions capture both spatial and temporal information in videos
- C3D is a simple, effective 3D CNN for video classification
- I3D inflates 2D filters to 3D, leveraging pretrained 2D models for video tasks

These models form the foundation for more advanced video understanding architectures. 