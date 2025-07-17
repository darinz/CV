# 02. K-Nearest Neighbor (KNN)

K-Nearest Neighbor (KNN) is a simple, intuitive algorithm for classification and regression. Here, we focus on its use for image classification.

## What is KNN?

KNN is a non-parametric method that makes predictions based on the $`K`$ closest training examples in the feature space.

- **Training:** Store all labeled training data.
- **Prediction:** For a new image, find the $`K`$ nearest neighbors and predict the majority class among them.

## Distance Metrics

To find the nearest neighbors, we need a way to measure distance between feature vectors.

- **L1 (Manhattan) Distance:**

  $`d(x_i, x_j) = \sum_{k} |x_{ik} - x_{jk}|`$

- **L2 (Euclidean) Distance:**

  $`d(x_i, x_j) = \sqrt{\sum_{k} (x_{ik} - x_{jk})^2}`$

## KNN Algorithm Steps

1. Store all training examples $`(x_i, y_i)`$.
2. For a test image $`x_{test}`$:
   - Compute distances to all training examples.
   - Select the $`K`$ closest examples.
   - Predict the majority class among these $`K`$ neighbors.

## Python Example: KNN for Image Classification

Here’s a simple implementation using NumPy:

```python
import numpy as np
from collections import Counter

def l2_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(x_train, y_train, x_test, K=3):
    distances = [l2_distance(x_test, x) for x in x_train]
    k_indices = np.argsort(distances)[:K]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Example usage:
x_train = np.random.rand(5, 3072)  # 5 training images, flattened
y_train = [0, 1, 1, 0, 2]          # Labels
x_test = np.random.rand(3072)      # Test image

pred = knn_predict(x_train, y_train, x_test, K=3)
print(f"Predicted label: {pred}")
```

## Advantages and Limitations

**Advantages:**
- Simple to implement and understand
- No training phase
- Naturally handles multi-class problems

**Limitations:**
- Slow at test time: must compute distance to all training examples
- Memory intensive: stores all data
- Sensitive to irrelevant features
- Suffers from the curse of dimensionality

## Summary

KNN is a foundational algorithm that provides a baseline for more advanced methods. It’s easy to implement and understand, making it a great starting point for learning about classification.

---

**Next:** [Linear Classifiers](03_Linear_Classifiers.md) 