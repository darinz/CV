# LSTM (Long Short-Term Memory)

Long Short-Term Memory (LSTM) networks are a type of RNN designed to address the vanishing gradient problem and capture long-term dependencies in sequences.

## Why LSTM?

Standard RNNs struggle to learn long-term dependencies due to vanishing/exploding gradients. LSTMs solve this by introducing a memory cell and gating mechanisms that control information flow.

## LSTM Architecture

An LSTM cell maintains a cell state $`C_t`$ and a hidden state $`h_t`$. It uses three gates:
- **Forget gate** $`f_t`$
- **Input gate** $`i_t`$
- **Output gate** $`o_t`$

### LSTM Equations

**Cell State Update:**
```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

**Hidden State:**
```math
h_t = o_t \odot \tanh(C_t)
```

**Gates:**
- Forget gate:
  ```math
  f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
  ```
- Input gate:
  ```math
  i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
  ```
- Output gate:
  ```math
  o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
  ```
- Candidate cell value:
  ```math
  \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
  ```

Where $`\sigma`$ is the sigmoid function, $`\odot`$ is elementwise multiplication, and $`[h_{t-1}, x_t]`$ denotes concatenation.

## LSTM Variants

### Peephole LSTM
Gates can also access the cell state:
```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t, C_{t-1}] + b_f)
i_t = \sigma(W_i \cdot [h_{t-1}, x_t, C_{t-1}] + b_i)
o_t = \sigma(W_o \cdot [h_{t-1}, x_t, C_t] + b_o)
```

### Coupled LSTM
Forget and input gates are coupled:
```math
f_t = 1 - i_t
```

## Python Example: LSTM Cell from Scratch

Below is a minimal LSTM cell implementation using NumPy:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        concat_size = hidden_size + input_size
        self.W_f = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_f = np.zeros((hidden_size, 1))
        self.W_i = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_i = np.zeros((hidden_size, 1))
        self.W_o = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_o = np.zeros((hidden_size, 1))
        self.W_C = np.random.randn(hidden_size, concat_size) * 0.1
        self.b_C = np.zeros((hidden_size, 1))

    def step(self, x_t, h_prev, C_prev):
        concat = np.vstack((h_prev, x_t))
        f_t = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        i_t = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        o_t = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        C_tilde = np.tanh(np.dot(self.W_C, concat) + self.b_C)
        C_t = f_t * C_prev + i_t * C_tilde
        h_t = o_t * np.tanh(C_t)
        return h_t, C_t

# Example usage:
np.random.seed(42)
lstm = LSTMCell(input_size=3, hidden_size=5)
x_t = np.random.randn(3, 1)
h_prev = np.zeros((5, 1))
C_prev = np.zeros((5, 1))
h_t, C_t = lstm.step(x_t, h_prev, C_prev)
print('Next hidden state:', h_t.ravel())
print('Next cell state:', C_t.ravel())
```

**Explanation:**
- Each gate is computed using the previous hidden state and current input.
- The cell state $`C_t`$ acts as memory, allowing gradients to flow more easily through time.
- The output $`h_t`$ is a filtered version of the cell state.

---

Next: [GRU (Gated Recurrent Unit)](03_GRU.md) 