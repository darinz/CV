# Regularization and Optimization

This module explores the essential techniques for preventing overfitting and efficiently training machine learning models through regularization and advanced optimization algorithms.

## Regularization

Regularization techniques help prevent overfitting by adding constraints or penalties to the model parameters, encouraging simpler solutions that generalize better to unseen data.

### L2 Regularization (Weight Decay)

L2 regularization adds a penalty term proportional to the squared magnitude of the weights:

```math
L_{total} = L_{data} + \frac{\lambda}{2} \sum_{i,j} W_{ij}^2
```

where $\lambda$ is the regularization strength and $W_{ij}$ are the model parameters.

#### Gradient Update

The gradient update becomes:

```math
\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{data}}{\partial W} + \lambda W
```

```math
W \leftarrow W - \alpha \left(\frac{\partial L_{data}}{\partial W} + \lambda W\right)
```

#### Geometric Interpretation

L2 regularization can be viewed as constraining the weights to lie within a sphere centered at the origin, preventing any single weight from becoming too large.

### L1 Regularization (Lasso)

L1 regularization adds a penalty term proportional to the absolute magnitude of the weights:

```math
L_{total} = L_{data} + \lambda \sum_{i,j} |W_{ij}|
```

#### Properties

- **Sparsity**: Encourages many weights to become exactly zero
- **Feature Selection**: Automatically performs feature selection
- **Non-differentiable**: Requires special handling at zero

#### Gradient Update

```math
\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{data}}{\partial W} + \lambda \cdot \text{sign}(W)
```

### Dropout

Dropout randomly sets a fraction of neurons to zero during training, preventing co-adaptation:

```math
y = f(W \cdot (x \odot m))
```

where $m \sim \text{Bernoulli}(p)$ is a mask with dropout probability $p$.

#### Training vs. Inference

- **Training**: Apply dropout with probability $p$
- **Inference**: Scale outputs by $(1-p)$ or use all neurons

#### Mathematical Justification

Dropout can be viewed as training an ensemble of $2^n$ sub-networks, where $n$ is the number of neurons.

### Early Stopping

Early stopping monitors validation performance and stops training when it starts to degrade:

```math
\text{patience} = \arg\min_{t} \{t : \text{val\_loss}(t) > \min_{i \leq t} \text{val\_loss}(i) + \epsilon\}
```

## Stochastic Gradient Descent (SGD)

SGD is the fundamental optimization algorithm for training neural networks, updating parameters using gradients computed on mini-batches.

### Basic Algorithm

```math
\theta_{t+1} = \theta_t - \alpha_t \nabla_\theta L(\theta_t, \mathcal{B}_t)
```

where:
- $\theta_t$ are the parameters at step $t$
- $\alpha_t$ is the learning rate
- $\mathcal{B}_t$ is the mini-batch at step $t$

### Convergence Properties

Under certain conditions, SGD converges to a local minimum:

```math
\mathbb{E}[\|\nabla L(\theta_t)\|^2] \leq \frac{C}{\sqrt{t}}
```

where $C$ is a constant depending on the problem.

### Mini-batch Size Trade-offs

- **Small batches**: More noise, better generalization, slower convergence
- **Large batches**: Less noise, faster convergence, potential overfitting
- **Optimal size**: Typically 32-256 for most problems

## Momentum

Momentum accelerates SGD by accumulating gradients from previous steps, helping escape local minima and flat regions.

### Classical Momentum

```math
v_{t+1} = \mu v_t - \alpha \nabla_\theta L(\theta_t)
```

```math
\theta_{t+1} = \theta_t + v_{t+1}
```

where $\mu \in [0,1]$ is the momentum coefficient.

### Nesterov Momentum

Nesterov momentum provides better convergence by evaluating the gradient at the predicted position:

```math
v_{t+1} = \mu v_t - \alpha \nabla_\theta L(\theta_t + \mu v_t)
```

```math
\theta_{t+1} = \theta_t + v_{t+1}
```

### Physical Analogy

Momentum can be understood as a ball rolling down a hill:
- **Velocity**: Accumulated gradient direction
- **Friction**: Momentum decay factor
- **Acceleration**: Current gradient

## AdaGrad

AdaGrad adapts the learning rate for each parameter based on the historical gradient magnitudes.

### Algorithm

```math
G_t = G_{t-1} + (\nabla_\theta L(\theta_t))^2
```

```math
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla_\theta L(\theta_t)
```

where $\epsilon$ is a small constant for numerical stability.

### Properties

- **Adaptive Learning Rates**: Parameters with large gradients get smaller learning rates
- **Sparse Features**: Works well for sparse data
- **Monotonic Decay**: Learning rates only decrease, potentially stopping learning too early

### Limitations

- **Aggressive Decay**: Learning rates can become very small
- **Memory Usage**: Requires storing gradient history for each parameter

## Adam

Adam combines the benefits of AdaGrad and RMSprop with momentum, providing adaptive learning rates with bias correction.

### Algorithm

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

### Hyperparameters

- $\alpha = 0.001$: Learning rate
- $\beta_1 = 0.9$: First moment decay
- $\beta_2 = 0.999$: Second moment decay
- $\epsilon = 10^{-8}$: Numerical stability

### Advantages

- **Adaptive Learning Rates**: Each parameter has its own learning rate
- **Bias Correction**: Handles initialization bias
- **Robust**: Works well across many problems
- **Memory Efficient**: Only requires first and second moment estimates

## Learning Rate Schedules

Learning rate scheduling adjusts the learning rate during training to improve convergence and final performance.

### Step Decay

Reduce learning rate by a factor at predetermined steps:

```math
\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}
```

where:
- $\alpha_0$ is the initial learning rate
- $\gamma$ is the decay factor (typically 0.1)
- $s$ is the step size

### Exponential Decay

Continuous exponential decay:

```math
\alpha_t = \alpha_0 \cdot e^{-kt}
```

where $k$ controls the decay rate.

### Cosine Annealing

Smooth cosine-based decay:

```math
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T}\pi))
```

where $T$ is the total number of steps.

### Warmup

Gradually increase learning rate at the beginning:

```math
\alpha_t = \alpha_{max} \cdot \min(1, \frac{t}{t_{warmup}})
```

### Cyclical Learning Rates

Oscillate between minimum and maximum learning rates:

```math
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T_{cycle}}\pi))
```

## Practical Considerations

### Hyperparameter Tuning

**Learning Rate:**
- Start with $\alpha = 0.001$ for Adam
- Use learning rate finder to determine optimal range
- Monitor loss curves for signs of instability

**Regularization Strength:**
- $\lambda = 0.0001$ to $0.01$ for L2 regularization
- Use validation set to tune $\lambda$
- Consider dataset size and model complexity

**Momentum:**
- $\mu = 0.9$ for most problems
- Higher values (0.95-0.99) for fine-tuning
- Lower values (0.5-0.7) for initial training

### Monitoring Training

**Key Metrics:**
- Training and validation loss
- Learning rate schedule
- Gradient norms
- Parameter norms

**Early Warning Signs:**
- Exploding gradients: $\|\nabla L\| > 10$
- Vanishing gradients: $\|\nabla L\| < 10^{-6}$
- Oscillating loss: Unstable learning rate

### Implementation Tips

1. **Gradient Clipping**: Prevent exploding gradients
2. **Weight Initialization**: Use proper initialization schemes
3. **Batch Normalization**: Often reduces need for aggressive regularization
4. **Data Augmentation**: Natural form of regularization
5. **Ensemble Methods**: Combine multiple models for better generalization

## Summary

Regularization and optimization are fundamental to training effective machine learning models:

1. **Regularization** prevents overfitting through various techniques
2. **SGD** provides the foundation for parameter updates
3. **Advanced Optimizers** (Momentum, AdaGrad, Adam) improve convergence
4. **Learning Rate Schedules** adapt the learning rate during training

The choice of regularization and optimization techniques depends on the specific problem, dataset characteristics, and computational constraints. Understanding these methods is crucial for building robust and efficient machine learning systems. 