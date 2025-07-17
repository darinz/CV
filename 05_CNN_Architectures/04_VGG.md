# 04. VGG

VGG, introduced in 2014, demonstrated that increasing network depth using small 3×3 convolutions can significantly improve performance. VGG-16 and VGG-19 are the most well-known variants.

## Architecture Design Principles

1. **Small Kernels:** 3×3 convolutions with padding=1
2. **Increasing Depth:** More layers for better feature learning
3. **Doubling Channels:** Channels double after each pooling layer
4. **MaxPooling:** 2×2 max pooling for spatial reduction

## VGG-16 Architecture

- **Input:** $224 \times 224 \times 3$
- **Blocks:**
  1. Conv(64, 3×3) → ReLU → Conv(64, 3×3) → ReLU → MaxPool(2×2)
     ```math
     \text{Output: } 112 \times 112 \times 64
     ```
  2. Conv(128, 3×3) → ReLU → Conv(128, 3×3) → ReLU → MaxPool(2×2)
     ```math
     \text{Output: } 56 \times 56 \times 128
     ```
  3. Conv(256, 3×3) → ReLU → Conv(256, 3×3) → ReLU → Conv(256, 3×3) → ReLU → MaxPool(2×2)
     ```math
     \text{Output: } 28 \times 28 \times 256
     ```
  4. Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → MaxPool(2×2)
     ```math
     \text{Output: } 14 \times 14 \times 512
     ```
  5. Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → MaxPool(2×2)
     ```math
     \text{Output: } 7 \times 7 \times 512
     ```
- **FC Layers:** FC(4096) → ReLU → Dropout(0.5) → FC(4096) → ReLU → Dropout(0.5) → FC(1000) → Softmax

## Mathematical Analysis

### Receptive Field
For 3×3 convolutions with padding=1:
```math
\text{RF}_l = \text{RF}_{l-1} + 2
```

### Parameters per Layer
```math
\text{Params} = k^2 \cdot c_{in} \cdot c_{out} + c_{out}
```
For 3×3 convolution:
```math
\text{Params} = 9 \cdot c_{in} \cdot c_{out} + c_{out}
```

## VGG Variants
- **VGG-11:** 8 conv + 3 FC layers
- **VGG-13:** 10 conv + 3 FC layers
- **VGG-16:** 13 conv + 3 FC layers
- **VGG-19:** 16 conv + 3 FC layers

## Python Example: VGG-16 in PyTorch

```python
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x
```

## Summary
- VGG showed the power of depth and small convolutions
- Simple, uniform architecture, but parameter-heavy
- Still used as a feature extractor in transfer learning

---

**Next:** [ResNet](05_ResNet.md) 