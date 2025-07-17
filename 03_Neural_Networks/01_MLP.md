# 01. Multi-layer Perceptron (MLP) and Architecture

A Multi-layer Perceptron (MLP) is a type of feedforward artificial neural network consisting of multiple layers of neurons. Each neuron in one layer is connected to every neuron in the next layer, allowing the network to learn complex, non-linear functions.

## Architecture

An MLP with $`L`$ layers has the following structure:

```math
\text{Input Layer: } x \in \mathbb{R}^{d_0}
```

```math
\text{Hidden Layers: } h^{(l)} \in \mathbb{R}^{d_l}, \quad l = 1, 2, \ldots, L-1
```

```math
\text{Output Layer: } y \in \mathbb{R}^{d_L}
```

where $`d_l`$ is the number of neurons in layer $`l`$.

## Forward Propagation

The forward pass computes the output through each layer:

```math
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
```

- $`W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}`$: Weight matrix for layer $`l`$
- $`b^{(l)} \in \mathbb{R}^{d_l}`$: Bias vector for layer $`l`$
- $`\sigma`$: Activation function
- $`h^{(0)} = x`$: Input layer

## Python Example: Forward Pass in an MLP

Here’s a simple implementation of a 2-layer MLP (1 hidden layer) using NumPy:

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# Network dimensions
input_dim = 4
hidden_dim = 5
output_dim = 3

# Randomly initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(hidden_dim, input_dim)
b1 = np.random.randn(hidden_dim)
W2 = np.random.randn(output_dim, hidden_dim)
b2 = np.random.randn(output_dim)

# Example input
x = np.random.randn(input_dim)

# Forward pass
h1 = relu(W1 @ x + b1)
y = softmax(W2 @ h1 + b2)
print("Output probabilities:", y)
```

## Why Use Multiple Layers?
- **Single-layer (linear) models** can only learn linear functions.
- **Multiple layers** allow the network to learn hierarchical, non-linear representations, making MLPs universal function approximators.

## Summary
- MLPs consist of input, hidden, and output layers.
- Each layer applies a linear transformation followed by a non-linear activation.
- Forward propagation computes the output step by step through the layers.

---

**Next:** [Activation Functions](02_Activation_Functions.md) 