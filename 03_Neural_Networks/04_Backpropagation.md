# 04. Backpropagation

Backpropagation is the core algorithm for efficiently computing gradients in neural networks. It enables training by propagating error signals backward through the network.

## Chain Rule Foundation

Backpropagation relies on the chain rule of calculus to compute gradients layer by layer:

```math
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial h^{(l)}} \frac{\partial h^{(l)}}{\partial W^{(l)}}
```

```math
\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial h^{(l)}} \frac{\partial h^{(l)}}{\partial b^{(l)}}
```

## Error Signal (Delta)

Define the error signal for layer $`l`$:

```math
\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}
```

where $`z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}`$ is the pre-activation.

## Backpropagation Algorithm Steps

### Step 1: Forward Pass

Compute activations for all layers:

```math
z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}
```

```math
h^{(l)} = \sigma(z^{(l)})
```

### Step 2: Output Layer Error

For the output layer $`L`$:

```math
\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial h^{(L)}} \odot \sigma'(z^{(L)})
```

For cross-entropy loss with softmax:

```math
\delta^{(L)} = h^{(L)} - t
```

where $`t`$ is the target vector.

### Step 3: Backpropagate Error

For layers $`l = L-1, L-2, \ldots, 1`$:

```math
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})
```

### Step 4: Compute Gradients

For each layer $`l`$:

```math
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T
```

```math
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
```

## Python Example: Backpropagation for a 2-layer MLP

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# Forward pass (see previous example)
# ...
# Assume h1, y, x, W1, W2, b1, b2, y_true are defined

# Backward pass
# Output layer error

delta2 = y - y_true  # for softmax + cross-entropy
# Gradients for W2, b2

dW2 = np.outer(delta2, h1)
db2 = delta2
# Hidden layer error

delta1 = (W2.T @ delta2) * relu_derivative(W1 @ x + b1)
# Gradients for W1, b1

dW1 = np.outer(delta1, x)
db1 = delta1
```

## Matrix Formulation (Mini-batch)

For batch size $`B`$:

```math
\frac{\partial L}{\partial W^{(l)}} = \frac{1}{B} \sum_{i=1}^{B} \delta_i^{(l)} (h_i^{(l-1)})^T
```

```math
\frac{\partial L}{\partial b^{(l)}} = \frac{1}{B} \sum_{i=1}^{B} \delta_i^{(l)}
```

## Summary
- Backpropagation efficiently computes gradients for all parameters
- Uses the chain rule to propagate error signals backward
- Enables gradient-based optimization for deep networks

---

**Next:** [Training Process](05_Training_Process.md) 