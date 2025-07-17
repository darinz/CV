# 03. Momentum

Momentum is an optimization technique that accelerates Stochastic Gradient Descent (SGD) by accumulating gradients from previous steps. It helps escape local minima and speeds up convergence, especially in regions with flat or noisy gradients.

## Classical Momentum

Momentum adds a velocity term $`v`$ to the parameter update:

```math
v_{t+1} = \mu v_t - \alpha \nabla_\theta L(\theta_t)
```

```math
\theta_{t+1} = \theta_t + v_{t+1}
```

- $`\mu`$: Momentum coefficient (typically 0.9)
- $`\alpha`$: Learning rate
- $`v_t`$: Velocity at step $`t`$

### Python Example: Classical Momentum

```python
import numpy as np

W = np.random.randn(3, 3072)
v = np.zeros_like(W)
mu = 0.9
alpha = 1e-3
for step in range(100):
    grad = np.random.randn(*W.shape)  # dummy gradient
    v = mu * v - alpha * grad
    W += v
```

## Nesterov Momentum

Nesterov momentum improves upon classical momentum by evaluating the gradient at the predicted next position:

```math
v_{t+1} = \mu v_t - \alpha \nabla_\theta L(\theta_t + \mu v_t)
```

```math
\theta_{t+1} = \theta_t + v_{t+1}
```

### Python Example: Nesterov Momentum

```python
for step in range(100):
    lookahead_W = W + mu * v
    grad = np.random.randn(*W.shape)  # dummy gradient at lookahead_W
    v = mu * v - alpha * grad
    W += v
```

## Physical Analogy

Momentum can be visualized as a ball rolling down a hill:
- **Velocity**: Accumulated gradient direction
- **Friction**: Momentum decay factor $`(1-\mu)`$
- **Acceleration**: Current gradient

## Summary
- **Classical Momentum**: Uses past gradients to smooth updates
- **Nesterov Momentum**: Looks ahead for more accurate updates
- **Benefit**: Faster convergence, helps escape local minima

---

**Next:** [AdaGrad](04_AdaGrad.md) 