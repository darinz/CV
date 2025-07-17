# GRU (Gated Recurrent Unit)

Gated Recurrent Units (GRUs) are a simplified variant of LSTM networks. They use fewer gates and parameters, making them computationally efficient while still addressing the vanishing gradient problem.

## Why GRU?

GRUs combine the cell and hidden state into a single state and use only two gates: update and reset. This makes them easier to implement and faster to train than LSTMs, while often achieving similar performance.

## GRU Architecture

A GRU cell uses:
- **Update gate** $`z_t`$
- **Reset gate** $`r_t`$

### GRU Equations

**Update gate:**
```math
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
```

**Reset gate:**
```math
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
```

**Candidate hidden state:**
```math
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
```

**Hidden state update:**
```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
```

Where $`\sigma`$ is the sigmoid function and $`\odot`$ is elementwise multiplication.

## Comparison: LSTM vs GRU

| Feature      | LSTM         | GRU         |
|-------------|--------------|-------------|
| Gates       | 3 (input, forget, output) | 2 (update, reset) |
| Cell state  | Yes          | No          |
| Complexity  | Higher       | Lower       |
| Performance | Similar      | Similar     |

## Python Example: GRU Cell from Scratch

Below is a minimal GRU cell implementation using NumPy:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        concat_size = hidden_size + input_size
        self.W_z = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_z = np.zeros((hidden_size, 1))
        self.W_r = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_r = np.zeros((hidden_size, 1))
        self.W_h = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_h = np.zeros((hidden_size, 1))

    def step(self, x_t, h_prev):
        concat = np.vstack((h_prev, x_t))
        z_t = sigmoid(np.dot(self.W_z, concat) + self.b_z)
        r_t = sigmoid(np.dot(self.W_r, concat) + self.b_r)
        concat_reset = np.vstack((r_t * h_prev, x_t))
        h_tilde = np.tanh(np.dot(self.W_h, concat_reset) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t

# Example usage:
np.random.seed(42)
gru = GRUCell(input_size=3, hidden_size=5)
x_t = np.random.randn(3, 1)
h_prev = np.zeros((5, 1))
h_t = gru.step(x_t, h_prev)
print('Next hidden state:', h_t.ravel())
```

**Explanation:**
- The update gate $`z_t`$ controls how much of the previous hidden state is kept.
- The reset gate $`r_t`$ controls how much of the previous hidden state to forget.
- The candidate hidden state $`\tilde{h}_t`$ is computed using the reset gate.
- The final hidden state $`h_t`$ is a blend of the previous hidden state and the candidate.

---

Next: [Language Modeling](04_Language_Modeling.md) 