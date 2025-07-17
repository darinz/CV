# 05. Adam Optimizer

Adam (Adaptive Moment Estimation) is a popular optimization algorithm that combines the benefits of AdaGrad and RMSprop with momentum. It adapts the learning rate for each parameter and includes bias correction.

## Adam Algorithm

Adam maintains two moving averages for each parameter:
- **First moment (mean):** $`m_t`$
- **Second moment (uncentered variance):** $`v_t`$

### Update Equations

**First Moment (Momentum):**
```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta L(\theta_t)
```

**Second Moment (Adaptive Learning Rate):**
```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta L(\theta_t))^2
```

**Bias Correction:**
```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```
```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

**Parameter Update:**
```math
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
```

- $`\alpha`$: Learning rate (default 0.001)
- $`\beta_1`$: Decay rate for first moment (default 0.9)
- $`\beta_2`$: Decay rate for second moment (default 0.999)
- $`\epsilon`$: Small constant for numerical stability (default $`10^{-8}`$)

## Python Example: Adam Optimizer

```python
import numpy as np

W = np.random.randn(3, 3072)
m = np.zeros_like(W)
v = np.zeros_like(W)
alpha = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
for t in range(1, 101):
    grad = np.random.randn(*W.shape)  # dummy gradient
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    W -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
```

## Advantages
- **Adaptive Learning Rates**: Each parameter has its own learning rate
- **Bias Correction**: Handles initialization bias
- **Robust**: Works well across many problems
- **Memory Efficient**: Only requires first and second moment estimates

## Summary
- Adam is widely used for training deep neural networks
- Combines momentum and adaptive learning rates
- Default hyperparameters work well for most problems

---

**Next:** [Learning Rate Schedules](06_Learning_Rate_Schedules.md) 