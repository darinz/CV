# 01. History and Development of CNNs

Convolutional Neural Networks (CNNs) have evolved from biological inspiration to become the dominant architecture for computer vision tasks. Understanding this history helps appreciate the design principles behind modern CNNs.

## Early Developments

### 1960s - Hubel and Wiesel
David Hubel and Torsten Wiesel discovered that neurons in the visual cortex respond to specific patterns and orientations. This biological insight inspired the concept of **receptive fields** in CNNs.

- **Simple cells:** Oriented edges and bars
- **Complex cells:** Movement and position invariance

### 1980s - Neocognitron
Kunihiko Fukushima developed the **Neocognitron**, a hierarchical neural network with:
- **Local receptive fields:** Neurons only respond to specific regions
- **Shared weights:** Same pattern detector applied across the image
- **Hierarchical structure:** Multiple layers for feature extraction

### 1989 - Backpropagation for CNNs
Yann LeCun successfully applied backpropagation to convolutional networks, creating **LeNet-5** for handwritten digit recognition.

## Modern Era

### 2012 - AlexNet
Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton achieved breakthrough performance on ImageNet, marking the beginning of the deep learning revolution.

**Key Innovations:**
- **ReLU activation:** Faster training than sigmoid/tanh
- **Dropout:** Regularization technique
- **GPU training:** Parallel computation
- **Data augmentation:** Improved generalization

### 2014 - VGGNet
Karen Simonyan and Andrew Zisserman introduced deeper networks with consistent 3×3 convolutions.

- **Small kernels:** 3×3 convolutions throughout
- **Deep structure:** 16 layers
- **Simple design:** Easy to understand and implement

### 2015 - ResNet
Kaiming He et al. introduced **skip connections** (residual connections), enabling training of very deep networks.

**Key Innovation:**
```math
h^{(l+1)} = \mathcal{F}(h^{(l)}) + h^{(l)}
```
This allows gradients to flow directly through skip connections, solving the vanishing gradient problem.

### 2017 - Transformer
Ashish Vaswani et al. introduced attention mechanisms, later adapted for vision tasks as Vision Transformers (ViT).

## Python Example: Simple CNN Evolution

Here's a simple example showing how CNN architectures evolved:

```python
import torch
import torch.nn as nn

# LeNet-5 style (1990s)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# AlexNet style (2012)
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

The history of CNNs shows a progression from:
1. **Biological inspiration** (Hubel & Wiesel)
2. **Early neural networks** (Neocognitron, LeNet)
3. **Deep learning revolution** (AlexNet)
4. **Modern architectures** (VGG, ResNet, Transformers)

Each milestone introduced key innovations that are still used in modern CNNs.

---

**Next:** [Higher-level Representations and Feature Learning](02_Feature_Learning.md) 