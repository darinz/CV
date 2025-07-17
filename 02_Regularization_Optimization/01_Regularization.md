# 01. Regularization

Regularization techniques help prevent overfitting by adding constraints or penalties to model parameters, encouraging simpler solutions that generalize better to unseen data. Here, we cover L2 regularization, L1 regularization, dropout, and early stopping.

## L2 Regularization (Weight Decay)

L2 regularization adds a penalty proportional to the squared magnitude of the weights:

```math
L_{total} = L_{data} + \frac{\lambda}{2} \sum_{i,j} W_{ij}^2
```

- $`L_{data}`$: Original loss (e.g., cross-entropy)
- $`\lambda`$: Regularization strength
- $`W_{ij}`$: Model parameters

### Gradient Update

The gradient update with L2 regularization is:

```math
\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{data}}{\partial W} + \lambda W
```

```math
W \leftarrow W - \alpha \left(\frac{\partial L_{data}}{\partial W} + \lambda W\right)
```

- $`\alpha`$: Learning rate

#### Python Example: L2 Regularization

```python
import numpy as np

W = np.random.randn(3, 3072)
loss_data = 1.0  # dummy loss
lambda_ = 0.01
alpha = 1e-3

grad_data = np.random.randn(*W.shape)  # dummy gradient
W -= alpha * (grad_data + lambda_ * W)
```

## L1 Regularization (Lasso)

L1 regularization adds a penalty proportional to the absolute value of the weights:

```math
L_{total} = L_{data} + \lambda \sum_{i,j} |W_{ij}|
```

### Properties
- **Sparsity**: Encourages many weights to become exactly zero
- **Feature Selection**: Automatically selects important features
- **Non-differentiable**: Special handling needed at zero

### Gradient Update

```math
\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{data}}{\partial W} + \lambda \cdot \text{sign}(W)
```

#### Python Example: L1 Regularization

```python
W = np.random.randn(3, 3072)
lambda_ = 0.01
grad_data = np.random.randn(*W.shape)
W -= alpha * (grad_data + lambda_ * np.sign(W))
```

## Dropout

Dropout randomly sets a fraction of neurons to zero during training, preventing co-adaptation and improving generalization.

```math
y = f(W \cdot (x \odot m))
```

- $`m \sim \text{Bernoulli}(p)`$: Mask with dropout probability $`p`$

### Training vs. Inference
- **Training**: Apply dropout with probability $`p`$
- **Inference**: Use all neurons, scale outputs by $`(1-p)`$

#### Python Example: Dropout

```python
np.random.seed(0)
x = np.random.randn(10)
p = 0.5  # dropout probability
mask = (np.random.rand(*x.shape) > p).astype(float)
x_dropout = x * mask / (1 - p)
```

## Early Stopping

Early stopping monitors validation performance and stops training when it starts to degrade, preventing overfitting.

```math
\text{patience} = \arg\min_{t} \{t : \text{val\_loss}(t) > \min_{i \leq t} \text{val\_loss}(i) + \epsilon\}
```

#### Python Example: Early Stopping (Conceptual)

```python
best_val_loss = float('inf')
patience, patience_counter = 5, 0
for epoch in range(epochs):
    # train ...
    val_loss = ...
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

## Summary
- **L2 Regularization**: Penalizes large weights, smooths solutions
- **L1 Regularization**: Promotes sparsity, feature selection
- **Dropout**: Prevents co-adaptation, acts as model ensemble
- **Early Stopping**: Stops training before overfitting

---

**Next:** [Stochastic Gradient Descent (SGD)](02_SGD.md) 