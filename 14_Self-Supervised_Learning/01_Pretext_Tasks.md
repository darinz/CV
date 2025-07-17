# Pretext Tasks in Self-Supervised Learning

Self-supervised learning (SSL) leverages pretext tasks to learn useful representations from unlabeled data. Pretext tasks are designed so that the data itself provides the supervision signal. Below, we explore several common pretext tasks in computer vision, with detailed explanations, mathematical formulations, and Python code examples.

---

## 1. Image Colorization

**Goal:** Predict the color channels of an image given only its grayscale version.

### Explanation
- The model receives a grayscale image as input and is trained to predict the original color image.
- This forces the model to learn semantic and structural information about objects and scenes.

### Mathematical Formulation
Given a grayscale image $x_{gray}$ and its color version $x_{color}$, the model $f$ predicts $\hat{x}_{color} = f(x_{gray})$. The loss is typically mean squared error (MSE):

$$
L = \|x_{color} - \hat{x}_{color}\|^2
$$

### Python Example (using PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1)  # Predict ab channels (Lab color space)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example usage:
# x_gray: (batch, 1, H, W), x_ab: (batch, 2, H, W)
model = ColorizationNet()
criterion = nn.MSELoss()
output = model(x_gray)
loss = criterion(output, x_ab)
```

---

## 2. Jigsaw Puzzle

**Goal:** Predict the correct arrangement of shuffled image patches.

### Explanation
- The image is divided into patches, shuffled, and the model predicts the correct order.
- This encourages the model to understand spatial relationships and object structure.

### Mathematical Formulation
Let $x$ be the image, $\pi$ a permutation, and $f$ the model. The model predicts $\hat{\pi} = f(x_{\pi})$ (classification over possible permutations):

$$
L = -\sum_{i} y_i \log \hat{y}_i
$$

### Python Example (using PyTorch)
```python
import torch
import torch.nn as nn

class JigsawNet(nn.Module):
    def __init__(self, num_permutations):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64 * 9 * 9, num_permutations)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# x_patches: (batch, 3, 9, 9), y_perm: (batch,)
model = JigsawNet(num_permutations=100)
criterion = nn.CrossEntropyLoss()
output = model(x_patches)
loss = criterion(output, y_perm)
```

---

## 3. Rotation Prediction

**Goal:** Predict the rotation angle applied to an image (e.g., 0°, 90°, 180°, 270°).

### Explanation
- The model is trained to classify which rotation (out of 4) was applied to the input image.
- This encourages learning of object orientation and global structure.

### Mathematical Formulation
Given image $x$ and rotation $r \in \{0, 90, 180, 270\}$, the model predicts $\hat{r} = f(\text{rotate}(x, r))$:

$$
L = -\sum_{i} y_i \log \hat{y}_i
$$

### Python Example (using PyTorch)
```python
import torch
import torch.nn as nn

class RotationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 32 * 32, 4)  # 4 rotations
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# x_rot: (batch, 3, 32, 32), y_rot: (batch,)
model = RotationNet()
criterion = nn.CrossEntropyLoss()
output = model(x_rot)
loss = criterion(output, y_rot)
```

---

## 4. Inpainting

**Goal:** Predict missing regions in an image.

### Explanation
- Random regions of the image are masked out, and the model is trained to reconstruct the missing parts.
- This requires understanding of context and semantics.

### Mathematical Formulation
Given image $x$ and mask $M$, the model predicts $\hat{x}_{masked} = f(x \odot (1-M))$:

$$
L = \|x_{masked} - \hat{x}_{masked}\|^2
$$

### Python Example (using PyTorch)
```python
import torch
import torch.nn as nn

class InpaintingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# x_masked: (batch, 3, H, W), x_true: (batch, 3, H, W)
model = InpaintingNet()
criterion = nn.MSELoss()
output = model(x_masked)
loss = criterion(output, x_true)
```

---

## 5. Temporal Order Verification

**Goal:** For videos, predict the correct temporal order of frames.

### Explanation
- The model receives shuffled frames and must predict if the order is correct or classify the permutation.
- This helps the model learn temporal dynamics and causality.

### Mathematical Formulation
Let $[x_1, x_2, x_3]$ be frames, $\pi$ a permutation, and $f$ the model. The model predicts $\hat{\pi} = f([x_{\pi(1)}, x_{\pi(2)}, x_{\pi(3)}])$:

$$
L = -\sum_{i} y_i \log \hat{y}_i
$$

### Python Example (using PyTorch)
```python
import torch
import torch.nn as nn

class TemporalOrderNet(nn.Module):
    def __init__(self, num_permutations):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, (3, 3, 3), padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 3 * 16 * 16, num_permutations)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# x_frames: (batch, 3, 3, 16, 16), y_perm: (batch,)
model = TemporalOrderNet(num_permutations=6)
criterion = nn.CrossEntropyLoss()
output = model(x_frames)
loss = criterion(output, y_perm)
```

---

These pretext tasks enable models to learn rich, transferable representations from unlabeled data, forming the foundation for many self-supervised learning approaches. 