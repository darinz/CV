# 05. Training Process

Training a neural network involves iteratively updating its parameters to minimize the loss function. This is typically done using mini-batch gradient descent and backpropagation.

## Parameter Updates

Parameters are updated using the gradients computed by backpropagation:

```math
W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}
```

```math
b^{(l)} \leftarrow b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}
```

where $`\alpha`$ is the learning rate.

## Mini-batch Training

Training is performed on small batches of data for efficiency and better generalization.

### Steps in Mini-batch Training
1. **Forward Pass:** Compute predictions for the mini-batch
2. **Compute Loss:** Calculate loss and gradients
3. **Backward Pass:** Backpropagate error signals
4. **Update Parameters:** Apply gradient updates
5. **Repeat:** Continue until convergence

### Python Example: Mini-batch Training Loop

```python
import numpy as np

# Assume X (N x D), y_true (N x C), W1, b1, W2, b2 are initialized
batch_size = 32
alpha = 1e-3
for epoch in range(epochs):
    indices = np.random.permutation(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y_true[batch_idx]
        # Forward pass, loss, backward pass, and parameter update as above
```

## Convergence Criteria

- **Loss Threshold:** $`L < \epsilon`$
- **Gradient Norm:** $`\|\nabla L\| < \epsilon`$
- **Validation Performance:** Monitor validation loss/accuracy
- **Maximum Iterations:** Stop after $`T`$ epochs

## Summary
- Training alternates between forward and backward passes
- Mini-batch training improves efficiency and generalization
- Convergence is monitored using loss, gradients, and validation performance

---

**Next:** [Implementation Considerations](06_Implementation_Considerations.md) 