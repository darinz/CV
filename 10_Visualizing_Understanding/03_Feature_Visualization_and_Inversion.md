# Feature Visualization and Inversion

Understanding what neural networks learn is crucial for interpretability and trust. This guide covers feature visualization, inversion, and saliency techniques, with detailed explanations, math, and Python code examples.

## 1. Overview

- **Feature Visualization**: Visualize what neurons or layers respond to by generating images that maximize their activation.
- **Feature Inversion**: Reconstruct input images from intermediate feature representations.
- **Saliency Maps**: Highlight input regions most influential for a prediction.

---

## 2. Feature Visualization

### 2.1 Activation Maximization

Find an input $x^*$ that maximizes the activation of a neuron or layer $l$:

```math
x^* = \arg\max_x f_l(x) - \lambda \|x\|^2
```
where $f_l(x)$ is the activation, and $\lambda$ is a regularization parameter.

**Python Example (Activation Maximization):**
```python
import torch
from torchvision import models
from torch.optim import Adam

model = models.vgg16(pretrained=True).features.eval()
layer_idx = 10  # Example: visualize 10th layer

x = torch.randn(1, 3, 224, 224, requires_grad=True)
optimizer = Adam([x], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    out = x
    for idx, layer in enumerate(model):
        out = layer(out)
        if idx == layer_idx:
            break
    loss = -out.norm() + 1e-4 * x.norm()
    loss.backward()
    optimizer.step()
```

### 2.2 Gradient Ascent

Update the input in the direction of the gradient to maximize activation:

```math
x_{t+1} = x_t + \alpha \nabla_x f_l(x_t)
```

**Python Example:**
(See above: the optimization loop uses gradient ascent.)

### 2.3 Regularization

Regularization helps produce interpretable visualizations.
- **L2 Regularization:** $R(x) = \lambda \|x\|^2$
- **Total Variation:** $R(x) = \lambda \sum_{i,j} \sqrt{(x_{i+1,j} - x_{i,j})^2 + (x_{i,j+1} - x_{i,j})^2}$
- **Frequency Penalty:** $R(x) = \lambda \|\mathcal{F}(x)\|^2$

---

## 3. Feature Inversion

Reconstruct an input $x^*$ that produces a given feature representation $f_l(x_0)$:

```math
x^* = \arg\min_x \|f_l(x) - f_l(x_0)\|^2 + R(x)
```

**Python Example (Feature Inversion):**
```python
# Given a target image x0, reconstruct an image from its features
x0 = ...  # Original image tensor
model = models.vgg16(pretrained=True).features.eval()
layer_idx = 10

with torch.no_grad():
    target_feat = x0
    for idx, layer in enumerate(model):
        target_feat = layer(target_feat)
        if idx == layer_idx:
            break

def get_features(x):
    out = x
    for idx, layer in enumerate(model):
        out = layer(out)
        if idx == layer_idx:
            break
    return out

x = torch.randn_like(x0, requires_grad=True)
optimizer = Adam([x], lr=0.1)

for i in range(200):
    optimizer.zero_grad()
    feat = get_features(x)
    loss = ((feat - target_feat)**2).mean() + 1e-4 * x.norm()
    loss.backward()
    optimizer.step()
```

---

## 4. Saliency Maps

Saliency maps visualize which pixels most affect the model's output.

### 4.1 Gradient-based Saliency

Compute the gradient of the class score with respect to the input:

```math
S_{ij} = \left|\frac{\partial f_c}{\partial x_{ij}}\right|
```

**Python Example (Saliency Map):**
```python
import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(pretrained=True).eval()
img = Image.open('image.jpg')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
x = preprocess(img).unsqueeze(0).requires_grad_()

output = model(x)
score, idx = output.max(1)
score.backward()
saliency = x.grad.abs().max(dim=1)[0]
```

### 4.2 Guided Backpropagation

Guided backpropagation modifies the backward pass to only propagate positive gradients.

```math
\frac{\partial f_c}{\partial x_{ij}} = \begin{cases}
\frac{\partial f_c}{\partial x_{ij}} & \text{if } \frac{\partial f_c}{\partial x_{ij}} > 0 \\
0 & \text{otherwise}
\end{cases}
```

**Python Example:**
- Requires custom backward hooks; see [this guide](https://github.com/utkuozbulak/pytorch-cnn-visualizations#guided-backpropagation) for implementation.

### 4.3 Grad-CAM

Grad-CAM uses gradients of the target class flowing into the last convolutional layer to produce a coarse localization map.

```math
\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial f_c}{\partial A_{ij}^k}
```
```math
L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
```

**Python Example (Grad-CAM):**
- See [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for a ready-to-use implementation.

---

## 5. References
- [Feature Visualization Distill](https://distill.pub/2017/feature-visualization/)
- [Saliency Maps Paper](https://arxiv.org/abs/1312.6034)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Guided Backpropagation Paper](https://arxiv.org/abs/1412.6806) 