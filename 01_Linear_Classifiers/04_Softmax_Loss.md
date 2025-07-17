# 04. Softmax Loss (Cross-Entropy Loss)

The softmax loss, also known as cross-entropy loss, is the standard loss function for multi-class classification problems. It combines the softmax function (to produce probabilities) with the cross-entropy loss (to measure prediction quality).

## Softmax Function

The softmax function converts raw class scores into probabilities that sum to 1:

```math
P(y = k | x) = \frac{e^{f_k(x)}}{\sum_{j=1}^{C} e^{f_j(x)}}
```

- $`f_k(x)`$ is the score for class $`k`$.
- $`C`$ is the number of classes.

## Cross-Entropy Loss

The cross-entropy loss for a single example is:

```math
L = -\log P(y = k | x) = -f_k(x) + \log \sum_{j=1}^{C} e^{f_j(x)}
```

- $`y = k`$ is the true class label.

## Gradient Computation

The gradients with respect to the parameters are:

```math
\frac{\partial L}{\partial W_k} = (P(y = k | x) - \mathbb{1}[y = k]) x
```

```math
\frac{\partial L}{\partial b_k} = P(y = k | x) - \mathbb{1}[y = k]
```

where $`\mathbb{1}[y = k]`$ is 1 if the true class is $`k`$, else 0.

## Python Example: Softmax and Cross-Entropy Loss

```python
import numpy as np

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
    return exp_scores / np.sum(exp_scores)

def cross_entropy_loss(scores, y_true):
    probs = softmax(scores)
    return -np.log(probs[y_true])

# Example usage:
scores = np.array([2.0, 1.0, 0.1])  # raw scores for 3 classes
y_true = 0  # correct class
probs = softmax(scores)
loss = cross_entropy_loss(scores, y_true)
print(f"Probabilities: {probs}")
print(f"Cross-entropy loss: {loss}")
```

## Properties of Softmax Loss

- **Convex:** The loss is convex with respect to the parameters.
- **Numerically Stable:** Use the log-sum-exp trick for stability.
- **Probabilistic Interpretation:** Outputs can be interpreted as class probabilities.

## Summary

- Softmax loss is the standard for multi-class classification.
- It combines probability output with a principled loss function.
- Efficient and widely used in neural networks and linear classifiers.

---

**Next:** [Implementation Considerations](05_Implementation_Considerations.md) 