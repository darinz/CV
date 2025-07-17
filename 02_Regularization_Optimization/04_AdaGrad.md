# 04. AdaGrad

AdaGrad is an adaptive learning rate optimization algorithm that adjusts the learning rate for each parameter based on the historical sum of squared gradients. It is especially useful for sparse data and features.

## AdaGrad Algorithm

At each step $`t`$, AdaGrad updates parameters as follows:

```math
G_t = G_{t-1} + (\nabla_\theta L(\theta_t))^2
```

```math
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla_\theta L(\theta_t)
```

- $`G_t`$: Accumulated sum of squared gradients (per parameter)
- $`\alpha`$: Initial learning rate
- $`\epsilon`$: Small constant for numerical stability

## Properties
- **Adaptive Learning Rates**: Parameters with large gradients get smaller learning rates
- **Sparse Features**: Works well for sparse data
- **Monotonic Decay**: Learning rates only decrease, which can stop learning too early

## Limitations
- **Aggressive Decay**: Learning rates can become very small, halting progress
- **Memory Usage**: Requires storing gradient history for each parameter

## Python Example: AdaGrad

```python
import numpy as np

W = np.random.randn(3, 3072)
G = np.zeros_like(W)
alpha = 1e-2
epsilon = 1e-8
for step in range(100):
    grad = np.random.randn(*W.shape)  # dummy gradient
    G += grad ** 2
    W -= alpha * grad / (np.sqrt(G) + epsilon)
```

## Summary
- AdaGrad adapts learning rates for each parameter
- Good for sparse data, but learning rates can decay too quickly
- Use with care for long training runs

---

**Next:** [Adam](05_Adam.md) 