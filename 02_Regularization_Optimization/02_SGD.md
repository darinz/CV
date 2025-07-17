# 02. Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) is the fundamental optimization algorithm for training neural networks and other machine learning models. It updates parameters using gradients computed on small batches of data.

## Basic Algorithm

At each step $`t`$, SGD updates the parameters $`\theta`$ as follows:

```math
\theta_{t+1} = \theta_t - \alpha_t \nabla_\theta L(\theta_t, \mathcal{B}_t)
```

- $`\theta_t`$: Parameters at step $`t`$
- $`\alpha_t`$: Learning rate
- $`\mathcal{B}_t`$: Mini-batch at step $`t`$
- $`L`$: Loss function

## Why Use Mini-batches?

- **Small batches**: More noise, better generalization, slower convergence
- **Large batches**: Less noise, faster convergence, potential overfitting
- **Typical size**: 32–256 for most problems

## Convergence Properties

Under certain conditions, SGD converges to a local minimum:

```math
\mathbb{E}[\|\nabla L(\theta_t)\|^2] \leq \frac{C}{\sqrt{t}}
```

where $`C`$ is a constant depending on the problem.

## Python Example: SGD Loop

```python
import numpy as np

# Dummy data: 100 samples, 3072 features
N, D = 100, 3072
X = np.random.randn(N, D)
y = np.random.randint(0, 3, size=N)
W = np.random.randn(3, D)
alpha = 1e-3
batch_size = 32

for epoch in range(10):
    indices = np.random.permutation(N)
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        # Compute gradients (dummy example)
        grad = np.random.randn(*W.shape)
        W -= alpha * grad
```

## Mini-batch Size Trade-offs

- **Small batches**: More frequent updates, more noise, better generalization
- **Large batches**: Smoother updates, faster computation, risk of overfitting

## Summary
- SGD is the backbone of neural network optimization
- Uses mini-batches for efficiency and generalization
- Learning rate and batch size are key hyperparameters

---

**Next:** [Momentum](03_Momentum.md) 