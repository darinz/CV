# 02. Higher-level Representations and Feature Learning

CNNs learn hierarchical representations of images, extracting features from low-level edges to high-level semantic concepts. This guide explains how feature learning works in CNNs.

## Hierarchical Feature Learning

CNNs build up representations through multiple layers:

- **Low-level features:** Edges, corners, textures
- **Mid-level features:** Shapes, patterns, object parts
- **High-level features:** Objects, scenes, semantics

### Feature Hierarchy
Each layer extracts increasingly abstract features:

```math
\text{Layer 1: } f_1(x) = \text{edge detectors, color blobs}
```
```math
\text{Layer 2: } f_2(f_1(x)) = \text{combinations of edges, textures}
```
```math
\text{Layer 3: } f_3(f_2(f_1(x))) = \text{object parts, shapes}
```
```math
\text{Layer L: } f_L(\ldots) = \text{semantic concepts, object classes}
```

### Receptive Field
The receptive field of a neuron is the region of the input that affects its output:

```math
\text{RF}_l = \text{RF}_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i
```
- $`k_l`$: Kernel size at layer $`l`$
- $`s_i`$: Stride at layer $`i`$

## Python Example: Visualizing Feature Maps

You can visualize feature maps to see what each layer is learning:

```python
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Load a pre-trained model
model = models.vgg16(pretrained=True)
model.eval()

# Get the first conv layer
first_conv = model.features[0]

# Example input: random image
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    feature_maps = first_conv(x)

# Visualize the first 6 feature maps
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.show()
```

## Summary
- CNNs learn hierarchical features, from edges to objects
- Each layer builds on the previous, extracting more abstract representations
- Feature maps can be visualized to understand what the network is learning

---

**Next:** [Convolution Operations](03_Convolution.md) 