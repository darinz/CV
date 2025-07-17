# Image Classification with Linear Classifiers

This module explores the fundamentals of image classification using linear classifiers, covering both theoretical foundations and practical implementations.

## The Data-Driven Approach

The data-driven approach to image classification relies on learning patterns from labeled training data rather than hand-crafting rules. This paradigm shift has revolutionized computer vision by enabling systems to automatically learn discriminative features from examples.

### Key Principles

- **Learning from Examples**: Instead of manually defining features, we learn them from data
- **Generalization**: The goal is to perform well on unseen test data
- **Representation**: Images are typically represented as high-dimensional feature vectors

For an image with dimensions $H \times W \times C$ (height, width, channels), we flatten it into a feature vector:

```math
x \in \mathbb{R}^{D}, \quad D = H \times W \times C
```

## K-Nearest Neighbor (KNN)

KNN is a simple yet effective non-parametric method for classification that serves as a baseline for understanding more sophisticated approaches.

### Algorithm

1. **Training**: Store all training examples $(x_i, y_i)$ in memory
2. **Prediction**: For a test image $x_{test}$:
   - Find the $K$ nearest training examples based on distance metric
   - Predict the majority class among these $K$ neighbors

### Distance Metrics

Common distance functions include:

- **L1 (Manhattan) Distance**: $d(x_i, x_j) = \sum_{k} |x_{ik} - x_{jk}|$
- **L2 (Euclidean) Distance**: $d(x_i, x_j) = \sqrt{\sum_{k} (x_{ik} - x_{jk})^2}$

### Advantages and Limitations

**Advantages:**
- Simple to understand and implement
- No training phase required
- Naturally handles multi-class problems

**Limitations:**
- Computationally expensive at test time: $O(N)$ for $N$ training examples
- Memory intensive: requires storing all training data
- Sensitive to irrelevant features
- Curse of dimensionality

## Linear Classifiers

Linear classifiers learn a linear decision boundary to separate different classes in the feature space.

### Mathematical Formulation

A linear classifier computes scores for each class using a weight matrix $W$ and bias vector $b$:

```math
f(x) = Wx + b
```

where:
- $W \in \mathbb{R}^{C \times D}$ is the weight matrix
- $b \in \mathbb{R}^{C}$ is the bias vector
- $C$ is the number of classes
- $D$ is the feature dimension

The predicted class is the one with the highest score:

```math
y_{pred} = \arg\max_i f_i(x)
```

### Training Objective

The goal is to find parameters $W$ and $b$ that minimize a loss function over the training data:

```math
\min_{W,b} \frac{1}{N} \sum_{i=1}^{N} L(f(x_i), y_i) + \lambda R(W)
```

where $L$ is the loss function and $R(W)$ is a regularization term.

## Algebraic Viewpoint

From an algebraic perspective, linear classifiers perform matrix operations to transform input features into class scores.

### Score Computation

For a single image $x$:

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

### Parameter Learning

The parameters are updated using gradient descent:

```math
W \leftarrow W - \alpha \frac{\partial L}{\partial W}
```

```math
b \leftarrow b - \alpha \frac{\partial L}{\partial b}
```

where $\alpha$ is the learning rate.

## Visual Viewpoint

Visually, linear classifiers create decision boundaries that are hyperplanes in the feature space.

### Decision Boundaries

For a binary classification problem, the decision boundary is defined by:

```math
W^T x + b = 0
```

This creates a hyperplane that separates the feature space into two regions.

### Multi-class Visualization

In multi-class scenarios, the decision boundaries form a Voronoi tessellation where each region corresponds to a class. The boundaries are defined by:

```math
W_i^T x + b_i = W_j^T x + b_j
```

for classes $i$ and $j$.

## Geometric Viewpoint

From a geometric perspective, linear classifiers can be understood in terms of projections and distances.

### Distance to Decision Boundary

The signed distance from a point $x$ to the decision boundary is:

```math
d(x) = \frac{W^T x + b}{\|W\|}
```

### Margin Maximization

The goal is to maximize the margin, which is the minimum distance from any training point to the decision boundary:

```math
\text{margin} = \min_{i} \frac{y_i(W^T x_i + b)}{\|W\|}
```

This leads to the optimization problem:

```math
\max_{W,b} \frac{1}{\|W\|} \quad \text{subject to} \quad y_i(W^T x_i + b) \geq 1 \quad \forall i
```

## Softmax Loss

The softmax loss (also known as cross-entropy loss) is the most commonly used loss function for multi-class classification.

### Softmax Function

The softmax function converts raw scores into probabilities:

```math
P(y = k | x) = \frac{e^{f_k(x)}}{\sum_{j=1}^{C} e^{f_j(x)}}
```

### Cross-Entropy Loss

The loss function is the negative log-likelihood:

```math
L = -\log P(y = k | x) = -f_k(x) + \log \sum_{j=1}^{C} e^{f_j(x)}
```

### Gradient Computation

The gradients with respect to the parameters are:

```math
\frac{\partial L}{\partial W_k} = (P(y = k | x) - \mathbb{1}[y = k]) x
```

```math
\frac{\partial L}{\partial b_k} = P(y = k | x) - \mathbb{1}[y = k]
```

where $\mathbb{1}[y = k]$ is the indicator function.

### Properties

- **Convex**: The loss function is convex with respect to the parameters
- **Numerically Stable**: Can be computed efficiently using log-sum-exp trick
- **Probabilistic Interpretation**: Outputs can be interpreted as class probabilities

## Implementation Considerations

### Data Preprocessing

- **Normalization**: Scale features to have zero mean and unit variance
- **Regularization**: Add L2 regularization to prevent overfitting
- **Bias Trick**: Often combine $W$ and $b$ into a single matrix by adding a constant feature

### Optimization

- **Stochastic Gradient Descent**: Update parameters using mini-batches
- **Learning Rate Scheduling**: Gradually decrease learning rate during training
- **Momentum**: Use momentum to accelerate convergence

### Evaluation Metrics

- **Accuracy**: Percentage of correctly classified examples
- **Precision/Recall**: For imbalanced datasets
- **Confusion Matrix**: Detailed breakdown of classification results

## Summary

Linear classifiers provide a fundamental building block for image classification, offering:

1. **Simplicity**: Easy to understand and implement
2. **Efficiency**: Fast training and inference
3. **Interpretability**: Clear decision boundaries
4. **Foundation**: Basis for more complex models like neural networks

While linear classifiers have limitations in handling non-linear decision boundaries, they serve as an excellent starting point for understanding machine learning concepts and provide strong baselines for many classification tasks. 