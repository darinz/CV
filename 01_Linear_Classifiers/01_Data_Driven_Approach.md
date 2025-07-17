# 01. Data-Driven Approach to Image Classification

Image classification has evolved from hand-crafted rules to data-driven learning. In this guide, we’ll explore what the data-driven approach means, how images are represented mathematically, and how this sets the stage for modern machine learning.

## What is the Data-Driven Approach?

Instead of manually defining rules or features for recognizing images, we let algorithms learn patterns directly from labeled data. This is the foundation of modern computer vision.

- **Learning from Examples:** Rather than telling the computer what a cat looks like, we show it many labeled images of cats and non-cats, and it learns the distinguishing features.
- **Generalization:** The goal is to perform well on new, unseen images—not just memorize the training set.

## Image Representation

Images are typically represented as high-dimensional feature vectors. For a color image of height $`H`$, width $`W`$, and $`C`$ color channels (e.g., RGB), the image is a 3D array. To use it in machine learning, we flatten it into a 1D vector:

```math
x \in \mathbb{R}^{D}, \quad D = H \times W \times C
```

- $`x`$ is the feature vector for one image.
- $`D`$ is the total number of features (pixels × channels).

### Example: Flattening an Image in Python

Suppose we have a $32 \times 32$ RGB image (3 channels):

```python
import numpy as np

# Example image: 32x32 RGB
H, W, C = 32, 32, 3
image = np.random.randint(0, 256, size=(H, W, C), dtype=np.uint8)

# Flatten to a vector
x = image.flatten()
print(f"Shape: {x.shape}")  # Output: (3072,)
```

## Why is Representation Important?

The way we represent data affects how well a model can learn. Flattening preserves all pixel information, but loses spatial structure. More advanced models (like CNNs) can exploit this structure, but linear classifiers treat the input as a vector.

## Summary

- The data-driven approach learns from labeled examples.
- Images are represented as high-dimensional vectors for machine learning.
- This representation is the starting point for algorithms like KNN and linear classifiers.

---

**Next:** [K-Nearest Neighbor (KNN)](02_KNN.md) 