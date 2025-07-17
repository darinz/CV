# 03. AlexNet

AlexNet, introduced in 2012, was the first deep CNN to win the ImageNet challenge, marking the beginning of the deep learning revolution in computer vision.

## Architecture Overview

- **Input:** $227 \times 227 \times 3$
- **Layers:**
  1. Conv(96, 11×11, stride=4) → ReLU → MaxPool(3×3, stride=2)
     ```math
     \text{Output: } 55 \times 55 \times 96
     ```
  2. Conv(256, 5×5, pad=2) → ReLU → MaxPool(3×3, stride=2)
     ```math
     \text{Output: } 27 \times 27 \times 256
     ```
  3. Conv(384, 3×3, pad=1) → ReLU
     ```math
     \text{Output: } 27 \times 27 \times 384
     ```
  4. Conv(384, 3×3, pad=1) → ReLU
     ```math
     \text{Output: } 27 \times 27 \times 384
     ```
  5. Conv(256, 3×3, pad=1) → ReLU → MaxPool(3×3, stride=2)
     ```math
     \text{Output: } 13 \times 13 \times 256
     ```
  6. FC(4096) → ReLU → Dropout(0.5)
     ```math
     \text{Output: } 4096
     ```
  7. FC(4096) → ReLU → Dropout(0.5)
     ```math
     \text{Output: } 4096
     ```
  8. FC(1000) → Softmax
     ```math
     \text{Output: } 1000
     ```

## Key Innovations

1. **ReLU Activation:** Faster training than sigmoid/tanh
2. **Dropout:** Regularization to prevent overfitting
3. **Data Augmentation:** Random crops, horizontal flips
4. **GPU Training:** First large-scale GPU implementation

## Parameter Calculation

```math
\text{Total Parameters} = \sum_{l} (k_l^2 \cdot c_{l-1} \cdot c_l + c_l)
```

For AlexNet:
```math
\text{Parameters} \approx 60 \text{ million}
```

## Python Example: AlexNet in PyTorch

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```

## Summary
- AlexNet demonstrated the power of deep CNNs for large-scale image classification
- Introduced key innovations still used today
- Marked the start of the modern deep learning era in vision

---

**Next:** [VGG](04_VGG.md) 