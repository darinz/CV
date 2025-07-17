# 03. Loss Function: Softmax and Cross-Entropy

The loss function measures how well the neural network's predictions match the true labels. For classification, the most common choice is the softmax activation with cross-entropy loss.

## Softmax Activation

The softmax function converts raw output scores into probabilities:

```math
y_i = \frac{e^{z_i}}{\sum_{j=1}^{d_L} e^{z_j}}
```

- $`z_i`$: Pre-activation output of the $`i`$-th neuron in the output layer
- $`d_L`$: Number of output classes

## Cross-Entropy Loss

The cross-entropy loss for a single example is:

```math
L = -\sum_{i=1}^{d_L} t_i \log(y_i)
```

- $`t_i`$: Target value (1 for correct class, 0 otherwise)
- $`y_i`$: Predicted probability for class $`i`$

## Python Example: Softmax and Cross-Entropy

```python
import numpy as np

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# Example usage
z = np.array([2.0, 1.0, 0.1])  # raw output scores
y_true = np.array([1, 0, 0])   # true class is 0
y_pred = softmax(z)
loss = cross_entropy(y_pred, y_true)
print("Predicted probabilities:", y_pred)
print("Cross-entropy loss:", loss)
```

## Why Softmax + Cross-Entropy?
- Softmax outputs valid probabilities for multi-class classification
- Cross-entropy penalizes confident but wrong predictions
- The combination is differentiable and works well with gradient-based optimization

## Summary
- Use softmax for output layer in classification
- Cross-entropy loss measures prediction quality
- Together, they provide a principled way to train neural networks for classification

---

**Next:** [Backpropagation](04_Backpropagation.md) 