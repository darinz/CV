# 03. Linear Classifiers

Linear classifiers are fundamental models in machine learning that separate classes using linear decision boundaries. This guide covers their mathematical formulation, different viewpoints, and practical implementation.

## What is a Linear Classifier?

A linear classifier predicts the class of an input by computing a weighted sum of its features and adding a bias. The class with the highest score is chosen.

### Mathematical Formulation

Given an input feature vector $`x \in \mathbb{R}^D`$, the score for each class is computed as:

```math
f(x) = Wx + b
```

- $`W \in \mathbb{R}^{C \times D}`$ is the weight matrix
- $`b \in \mathbb{R}^C`$ is the bias vector
- $`C`$ is the number of classes
- $`D`$ is the feature dimension

The predicted class is:

```math
y_{pred} = \arg\max_i f_i(x)
```

### Python Example: Linear Classifier Score Computation

```python
import numpy as np

# Example: 3 classes, 3072 features (e.g., 32x32x3 image)
C, D = 3, 3072
W = np.random.randn(C, D)
b = np.random.randn(C)
x = np.random.randn(D)

scores = W @ x + b
predicted_class = np.argmax(scores)
print(f"Scores: {scores}")
print(f"Predicted class: {predicted_class}")
```

## Algebraic Viewpoint

Linear classifiers use matrix multiplication to transform input features into class scores. For a single image $`x`$:

```math
\begin{bmatrix}
s_1 \\
s_2 \\
\vdots \\
s_C
\end{bmatrix}
=
\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1D} \\
w_{21} & w_{22} & \cdots & w_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{C1} & w_{C2} & \cdots & w_{CD}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_D
\end{bmatrix}
+
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_C
\end{bmatrix}
```

## Visual Viewpoint

A linear classifier creates decision boundaries that are hyperplanes in the feature space.

- **Binary classification:** The decision boundary is:

  ```math
  W^T x + b = 0
  ```

- **Multi-class:** Each class has its own hyperplane; the space is divided into regions (Voronoi tessellation).

## Geometric Viewpoint

The signed distance from a point $`x`$ to the decision boundary is:

```math
d(x) = \frac{W^T x + b}{\|W\|}
```

- The margin is the minimum distance from any training point to the decision boundary.

### Margin Maximization

```math
\text{margin} = \min_{i} \frac{y_i(W^T x_i + b)}{\|W\|}
```

## Training Linear Classifiers

Parameters $`W`$ and $`b`$ are learned by minimizing a loss function over the training data, often using gradient descent.

### Parameter Update (Gradient Descent)

```math
W \leftarrow W - \alpha \frac{\partial L}{\partial W}
```

```math
b \leftarrow b - \alpha \frac{\partial L}{\partial b}
```

where $`\alpha`$ is the learning rate.

## Python Example: Training a Linear Classifier (Gradient Descent)

```python
# Dummy data: 10 images, 3072 features, 3 classes
N, D, C = 10, 3072, 3
X = np.random.randn(N, D)
y = np.random.randint(0, C, size=N)
W = np.random.randn(C, D)
b = np.random.randn(C)

# Simple loss: sum of squared errors (for illustration)
def loss(W, b, X, y):
    scores = X @ W.T + b
    correct_scores = scores[np.arange(N), y]
    return np.sum((scores - correct_scores[:, None]) ** 2)

# Gradient descent step
alpha = 1e-5
for step in range(100):
    # Compute gradients (dummy, not real gradients)
    dW = np.random.randn(C, D)
    db = np.random.randn(C)
    W -= alpha * dW
    b -= alpha * db
```

## Summary

- Linear classifiers use weighted sums to separate classes with hyperplanes.
- They are efficient, interpretable, and form the basis for more complex models.

---

**Next:** [Softmax Loss](04_Softmax_Loss.md) 