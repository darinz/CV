# 05. Implementation Considerations

When building linear classifiers for image classification, several practical considerations can greatly affect performance. This guide covers data preprocessing, optimization strategies, and evaluation metrics.

## Data Preprocessing

### Normalization

Scaling features to have zero mean and unit variance helps models converge faster and perform better.

```python
import numpy as np

# X: N x D matrix (N samples, D features)
X = np.random.rand(100, 3072)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-8  # avoid division by zero
X_norm = (X - X_mean) / X_std
```

### Regularization

Regularization helps prevent overfitting by penalizing large weights. L2 regularization is common:

```math
R(W) = \lambda \|W\|^2
```

- $`\lambda`$ is the regularization strength.

### Bias Trick

Combine $`W`$ and $`b`$ into a single matrix by adding a constant feature (usually 1) to each input vector.

## Optimization

### Stochastic Gradient Descent (SGD)

SGD updates parameters using small batches of data, making training faster and more scalable.

```python
# Simple SGD loop
learning_rate = 1e-3
batch_size = 32
for epoch in range(epochs):
    indices = np.random.permutation(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        # Compute gradients and update W, b (not shown)
```

### Learning Rate Scheduling

Gradually decreasing the learning rate can help the model converge to a better solution.

### Momentum

Momentum accelerates SGD by smoothing updates:

```python
v = 0
momentum = 0.9
for step in range(100):
    grad = ...  # compute gradient
    v = momentum * v - learning_rate * grad
    W += v
```

## Evaluation Metrics

### Accuracy

The percentage of correctly classified examples:

```python
accuracy = np.mean(y_pred == y_true)
```

### Precision, Recall, and Confusion Matrix

Useful for imbalanced datasets. Scikit-learn provides easy-to-use functions:

```python
from sklearn.metrics import precision_score, recall_score, confusion_matrix

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
cm = confusion_matrix(y_true, y_pred)
```

## Summary

- Normalize data and use regularization to improve generalization.
- Use SGD and learning rate schedules for efficient optimization.
- Evaluate with accuracy and other metrics for a complete picture of performance.

---

**End of Linear Classifiers Module** 