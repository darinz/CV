# 06. Learning Rate Schedules

Learning rate schedules adjust the learning rate during training to improve convergence and final model performance. Different schedules are suited for different tasks and training regimes.

## Step Decay

Reduce the learning rate by a factor at predetermined steps:

```math
\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}
```

- $`\alpha_0`$: Initial learning rate
- $`\gamma`$: Decay factor (e.g., 0.1)
- $`s`$: Step size

### Python Example: Step Decay

```python
alpha_0 = 0.1
gamma = 0.1
s = 10
for t in range(50):
    alpha_t = alpha_0 * (gamma ** (t // s))
    print(f"Step {t}: learning rate = {alpha_t}")
```

## Exponential Decay

Learning rate decays continuously:

```math
\alpha_t = \alpha_0 \cdot e^{-kt}
```

- $`k`$: Decay rate

### Python Example: Exponential Decay

```python
import numpy as np
alpha_0 = 0.1
k = 0.05
for t in range(50):
    alpha_t = alpha_0 * np.exp(-k * t)
    print(f"Step {t}: learning rate = {alpha_t}")
```

## Cosine Annealing

Smooth, periodic decay:

```math
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T}\pi))
```

- $`T`$: Total number of steps

### Python Example: Cosine Annealing

```python
import numpy as np
alpha_min = 0.001
alpha_max = 0.1
T = 50
for t in range(T):
    alpha_t = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t / T))
    print(f"Step {t}: learning rate = {alpha_t}")
```

## Warmup

Gradually increase the learning rate at the start of training:

```math
\alpha_t = \alpha_{max} \cdot \min(1, \frac{t}{t_{warmup}})
```

### Python Example: Warmup

```python
alpha_max = 0.1
t_warmup = 5
for t in range(20):
    alpha_t = alpha_max * min(1, t / t_warmup)
    print(f"Step {t}: learning rate = {alpha_t}")
```

## Cyclical Learning Rates

Oscillate between minimum and maximum learning rates:

```math
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T_{cycle}}\pi))
```

## Summary
- Learning rate schedules help models converge faster and avoid poor local minima
- Choose a schedule based on your dataset and model
- Combine with warmup for best results in deep learning

---

**Next:** [Practical Considerations](07_Practical_Considerations.md) 