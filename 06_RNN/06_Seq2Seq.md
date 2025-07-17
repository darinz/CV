# Sequence-to-Sequence (Seq2Seq) Models

Sequence-to-sequence (Seq2Seq) models map input sequences to output sequences of potentially different lengths. They are widely used in machine translation, summarization, and more.

## Encoder-Decoder Architecture

A Seq2Seq model consists of:
- **Encoder:** Processes the input sequence and summarizes it into a context vector.
- **Decoder:** Generates the output sequence, conditioned on the context vector.

### Encoder

The encoder processes the input $`x_1, x_2, \ldots, x_T`$:

```math
h_t^{(enc)} = \text{RNN}^{(enc)}(h_{t-1}^{(enc)}, x_t)
```

### Decoder

The decoder generates the output $`y_1, y_2, \ldots, y_{T_y}`$:

```math
h_t^{(dec)} = \text{RNN}^{(dec)}(h_{t-1}^{(dec)}, [y_{t-1}, c_t])
```

```math
P(y_t | y_1, \ldots, y_{t-1}, x) = \text{softmax}(W_{out} h_t^{(dec)} + b_{out})
```

Where $`c_t`$ is the context vector.

## Attention Mechanism

Attention allows the decoder to focus on different parts of the input sequence at each time step.

### Global Attention

```math
e_{t,i} = \text{score}(h_t^{(dec)}, h_i^{(enc)})
```

```math
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}
```

```math
c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i^{(enc)}
```

### Scoring Functions

- **Dot Product:**
  ```math
  \text{score}(h_t^{(dec)}, h_i^{(enc)}) = h_t^{(dec)} \cdot h_i^{(enc)}
  ```
- **General:**
  ```math
  \text{score}(h_t^{(dec)}, h_i^{(enc)}) = h_t^{(dec)} W_a h_i^{(enc)}
  ```
- **Concat:**
  ```math
  \text{score}(h_t^{(dec)}, h_i^{(enc)}) = v_a^T \tanh(W_a [h_t^{(dec)}; h_i^{(enc)}])
  ```

## Training

### Loss Function

```math
L = -\sum_{t=1}^{T_y} \log P(y_t | y_1, \ldots, y_{t-1}, x)
```

### Beam Search

During inference, beam search maintains the top-$`k`$ hypotheses:

```math
\mathcal{H}_t = \text{top-k}(\{h \oplus y_t : h \in \mathcal{H}_{t-1}, y_t \in V\})
```

Where $`\oplus`$ denotes concatenation.

## Python Example: Seq2Seq with Attention (Simplified)

Below is a simplified example of a Seq2Seq model with dot-product attention. This is for illustration and omits training:

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

def dot_attention(dec_h, enc_hs):
    # dec_h: (hidden_size, 1), enc_hs: (T, hidden_size, 1)
    scores = np.array([np.dot(dec_h.T, h)[0,0] for h in enc_hs])
    alphas = softmax(scores)
    context = sum(a * h for a, h in zip(alphas, enc_hs))
    return context, alphas

class SimpleSeq2Seq:
    def __init__(self, input_size, output_size, hidden_size):
        self.W_enc = np.random.randn(hidden_size, input_size) * 0.1
        self.W_dec = np.random.randn(hidden_size, output_size + hidden_size) * 0.1
        self.W_out = np.random.randn(output_size, hidden_size) * 0.1
        self.b_enc = np.zeros((hidden_size, 1))
        self.b_dec = np.zeros((hidden_size, 1))
        self.b_out = np.zeros((output_size, 1))

    def encode(self, inputs):
        hs = []
        h = np.zeros((self.W_enc.shape[0], 1))
        for x in inputs:
            h = np.tanh(np.dot(self.W_enc, x) + self.b_enc)
            hs.append(h)
        return hs

    def decode(self, outputs, enc_hs):
        h = np.zeros((self.W_dec.shape[0], 1))
        preds = []
        for y_prev in outputs:
            context, _ = dot_attention(h, enc_hs)
            inp = np.vstack((y_prev, context))
            h = np.tanh(np.dot(self.W_dec, inp) + self.b_dec)
            y = np.dot(self.W_out, h) + self.b_out
            preds.append(y)
        return preds

# Example usage:
np.random.seed(42)
inputs = [np.random.randn(3, 1) for _ in range(4)]
outputs = [np.random.randn(2, 1) for _ in range(3)]
model = SimpleSeq2Seq(input_size=3, output_size=2, hidden_size=5)
enc_hs = model.encode(inputs)
preds = model.decode(outputs, enc_hs)
print('Predicted outputs:', [y.ravel() for y in preds])
```

**Explanation:**
- The encoder processes the input sequence into hidden states.
- The decoder generates the output sequence, attending to the encoder's hidden states at each step.
- The attention mechanism computes a context vector as a weighted sum of encoder states.

---

Next: [Advanced RNN Architectures](07_Advanced_RNNs.md) 