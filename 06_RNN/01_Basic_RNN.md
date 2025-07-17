# Basic Recurrent Neural Networks (RNN)

Recurrent Neural Networks (RNNs) are designed for sequential data, where the order of data points matters (e.g., time series, text, speech).

## What is an RNN?

An RNN processes a sequence one element at a time, maintaining a hidden state that captures information about previous elements.

## Mathematical Formulation

For a sequence of inputs $`x_1, x_2, \ldots, x_T`$:

```math
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
```

```math
y_t = \sigma(W_{hy} h_t + b_y)
```

Where:
- $`h_t`$: hidden state at time $`t`$
- $`x_t`$: input at time $`t`$
- $`y_t`$: output at time $`t`$
- $`W_{hh}, W_{xh}, W_{hy}`$: weight matrices
- $`b_h, b_y`$: bias vectors
- $`\sigma`$: activation function (e.g., $`\tanh`$ or $`\text{ReLU}`$)

## Unrolling an RNN

An RNN can be visualized as a chain of repeating modules:

```math
h_1 = \sigma(W_{hh} h_0 + W_{xh} x_1 + b_h)
h_2 = \sigma(W_{hh} h_1 + W_{xh} x_2 + b_h)
\vdots
h_T = \sigma(W_{hh} h_{T-1} + W_{xh} x_T + b_h)
```

## Loss Function

For sequence prediction:

```math
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
```

Where $`L_t`$ is the loss at time $`t`$ (e.g., cross-entropy for classification).

## Backpropagation Through Time (BPTT)

To train an RNN, we use BPTT, which unfolds the network in time and applies backpropagation.

The gradient with respect to $`W_{hh}`$:

```math
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}
```

Where:

```math
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W_{hh}^T \text{diag}(\sigma'(h_i))
```

## Vanishing/Exploding Gradient Problem

The repeated multiplication in BPTT can cause gradients to vanish (become very small) or explode (become very large):

```math
\left\|\frac{\partial h_t}{\partial h_k}\right\| \leq \|W_{hh}\|^{t-k} \|\sigma'\|^{t-k}
```

- If $`\|W_{hh}\| < 1`$: gradients vanish
- If $`\|W_{hh}\| > 1`$: gradients explode

This makes training deep or long RNNs difficult.

## Python Example: Simple RNN from Scratch

Below is a minimal RNN implementation for sequence prediction using NumPy:

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def step(self, x_t, h_prev):
        h_t = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h)
        y_t = np.dot(self.W_hy, h_t) + self.b_y
        return h_t, y_t

    def forward(self, inputs):
        h = np.zeros((self.W_hh.shape[0], 1))
        hs, ys = [], []
        for x_t in inputs:
            h, y = self.step(x_t, h)
            hs.append(h)
            ys.append(y)
        return hs, ys

# Example usage:
np.random.seed(42)
rnn = SimpleRNN(input_size=3, hidden_size=5, output_size=2)
inputs = [np.random.randn(3, 1) for _ in range(4)]
hs, ys = rnn.forward(inputs)
print('Hidden states:', [h.ravel() for h in hs])
print('Outputs:', [y.ravel() for y in ys])
```

**Explanation:**
- `step` computes the next hidden state and output.
- `forward` processes a sequence of inputs.
- This code does not include training (backpropagation), but demonstrates the forward pass and hidden state evolution.

---

Next: [LSTM (Long Short-Term Memory)](02_LSTM.md) 