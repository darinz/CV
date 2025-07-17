# 02. Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Here are the most common activation functions used in MLPs.

## Sigmoid Function

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

- **Range:** $(0, 1)$
- **Derivative:** $`\sigma'(x) = \sigma(x)(1 - \sigma(x))`$
- **Vanishing Gradient Problem:** Derivative approaches zero for large $|x|$

### Python Example: Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

## Tanh Function

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

- **Range:** $(-1, 1)$
- **Derivative:** $`\tanh'(x) = 1 - \tanh^2(x)`$
- **Zero-centered:** Helps with optimization

### Python Example: Tanh

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
```

## ReLU (Rectified Linear Unit)

```math
\text{ReLU}(x) = \max(0, x)
```

- **Range:** $[0, \infty)$
- **Derivative:**

  $`
  \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
  `$ 
- **Advantages:** Simple, fast, helps with vanishing gradient problem
- **Disadvantages:** Dying ReLU problem (neurons can become inactive)

### Python Example: ReLU

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

## Leaky ReLU

```math
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
```

- $`\alpha`$ is a small positive constant (typically 0.01)
- **Fixes dying ReLU by allowing a small gradient when $x \leq 0$**

### Python Example: Leaky ReLU

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx
```

## Summary
- Activation functions introduce non-linearity
- Choice of activation affects training dynamics and performance
- ReLU is most common for hidden layers; softmax is used for output in classification

---

**Next:** [Loss Function](03_Loss_Function.md) 