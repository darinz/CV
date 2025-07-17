# Modern CNN Architectures

Modern convolutional neural network (CNN) architectures have evolved to address the limitations of early models, improve accuracy, and enable deeper networks. Here, we cover some of the most influential architectures and their key innovations.

---

## 1. AlexNet

**Key Innovations:**
- Introduced ReLU activation for faster training.
- Used dropout for regularization.
- Employed data augmentation and GPU training.

**Architecture Overview:**
- 5 convolutional layers, 3 fully connected layers.
- Max pooling after some conv layers.
- ReLU activations.

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ... more conv layers ...
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # ... more fc layers ...
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

---

## 2. VGGNet

**Key Innovations:**
- Used small (3x3) convolution filters.
- Increased depth (16-19 layers).

**Architecture Overview:**
- Stacks of 3x3 conv layers, followed by max pooling.
- Simpler, uniform design.

```python
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)
```

---

## 3. GoogLeNet (Inception)

**Key Innovations:**
- Inception modules: parallel conv layers with different filter sizes.
- 1x1 convolutions for dimensionality reduction.

**Inception Module:**
```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], 1)
```

---

## 4. ResNet

**Key Innovations:**
- Residual connections (skip connections) to enable very deep networks.
- Alleviates vanishing gradient problem.

**Residual Block:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
```

---

## 5. DenseNet

**Key Innovations:**
- Dense connections: each layer receives input from all previous layers.
- Improves feature reuse and gradient flow.

**Dense Block:**
```python
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)
```

---

## 6. Key Takeaways

- Modern CNNs use deeper architectures, skip connections, and modular designs.
- Innovations like batch normalization, dropout, and advanced optimizers are standard.
- Transfer learning with pre-trained models is common for practical applications.