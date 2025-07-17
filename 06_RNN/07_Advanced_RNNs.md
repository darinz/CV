# Advanced RNN Architectures

Advanced RNN architectures extend the basic RNN to improve performance on complex sequential tasks. Key variants include bidirectional RNNs, deep (stacked) RNNs, and RNNs with attention.

## Bidirectional RNN

A bidirectional RNN processes the sequence in both forward and backward directions, capturing information from past and future contexts.

### Mathematical Formulation

```math
h_t^{(f)} = \text{RNN}^{(f)}(h_{t-1}^{(f)}, x_t)
h_t^{(b)} = \text{RNN}^{(b)}(h_{t+1}^{(b)}, x_t)
h_t = [h_t^{(f)}; h_t^{(b)}]
```

Where $`[\cdot;\cdot]`$ denotes concatenation.

### Python Example: Bidirectional RNN (Forward Pass)

```python
import numpy as np

class SimpleBiRNN:
    def __init__(self, input_size, hidden_size):
        self.W_f = np.random.randn(hidden_size, input_size) * 0.1
        self.W_b = np.random.randn(hidden_size, input_size) * 0.1
        self.b_f = np.zeros((hidden_size, 1))
        self.b_b = np.zeros((hidden_size, 1))

    def forward(self, inputs):
        T = len(inputs)
        h_f = [np.zeros((self.W_f.shape[0], 1))]
        h_b = [np.zeros((self.W_b.shape[0], 1))]
        # Forward
        for x in inputs:
            h_f.append(np.tanh(np.dot(self.W_f, x) + self.b_f))
        # Backward
        for x in reversed(inputs):
            h_b.append(np.tanh(np.dot(self.W_b, x) + self.b_b))
        h_b = h_b[::-1]
        # Concatenate
        h = [np.vstack((hf, hb)) for hf, hb in zip(h_f[1:], h_b[1:])]
        return h

# Example usage:
inputs = [np.random.randn(3, 1) for _ in range(4)]
model = SimpleBiRNN(input_size=3, hidden_size=2)
hs = model.forward(inputs)
print('Bidirectional hidden states:', [h.ravel() for h in hs])
```

## Deep (Stacked) RNN

A deep RNN stacks multiple RNN layers, allowing the model to learn hierarchical representations.

### Mathematical Formulation

```math
h_t^{(l)} = \text{RNN}^{(l)}(h_{t-1}^{(l)}, h_t^{(l-1)})
```

Where $`l`$ is the layer index.

### Python Example: Deep RNN (Forward Pass)

```python
class SimpleDeepRNN:
    def __init__(self, input_size, hidden_sizes):
        self.layers = []
        for i, h_size in enumerate(hidden_sizes):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            W = np.random.randn(h_size, in_size) * 0.1
            b = np.zeros((h_size, 1))
            self.layers.append((W, b))

    def forward(self, inputs):
        hs = [inputs]
        for W, b in self.layers:
            out = []
            for x in hs[-1]:
                out.append(np.tanh(np.dot(W, x) + b))
            hs.append(out)
        return hs[1:]

# Example usage:
inputs = [np.random.randn(3, 1) for _ in range(4)]
model = SimpleDeepRNN(input_size=3, hidden_sizes=[4, 2])
hs = model.forward(inputs)
print('Layer 1 hidden states:', [h.ravel() for h in hs[0]])
print('Layer 2 hidden states:', [h.ravel() for h in hs[1]])
```

## Attention RNN

An RNN with attention can focus on different parts of the input sequence at each time step.

### Mathematical Formulation

```math
c_t = \text{Attention}(h_{t-1}, \{h_1, h_2, \ldots, h_T\})
h_t = \text{RNN}(h_{t-1}, [x_t, c_t])
```

Where $`c_t`$ is the context vector computed by the attention mechanism.

---

Next: [Training Techniques](08_Training_Techniques.md) 