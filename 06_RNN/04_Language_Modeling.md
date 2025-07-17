# Language Modeling with RNNs

Language modeling is the task of predicting the probability of the next word in a sequence, given the previous words. RNNs are well-suited for this because they can process sequences of arbitrary length.

## What is a Language Model?

A language model assigns a probability to a sequence of words $`w_1, w_2, \ldots, w_T`$:

```math
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t | w_1, w_2, \ldots, w_{t-1})
```

## RNN Language Model

At each time step, the RNN takes the embedding of the current word and the previous hidden state to produce a new hidden state and predict the next word:

```math
h_t = \text{RNN}(h_{t-1}, x_t)
```

```math
P(w_t | w_1, \ldots, w_{t-1}) = \text{softmax}(W_{out} h_t + b_{out})
```

Where $`x_t`$ is the embedding of $`w_t`$.

## Loss Function

The standard loss for language modeling is cross-entropy:

```math
L = -\sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})
```

## Perplexity

Perplexity measures how well a model predicts a sequence:

```math
\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(w_t | w_1, \ldots, w_{t-1})\right)
```

Lower perplexity indicates better performance.

## Word Embeddings

Words are represented as dense vectors (embeddings):

```math
x_t = E_{w_t}
```

Where $`E \in \mathbb{R}^{|V| \times d}`$ is the embedding matrix and $`|V|`$ is the vocabulary size.

## Python Example: RNN Language Model (Forward Pass)

Below is a simple RNN language model using NumPy. This example demonstrates the forward pass and prediction (no training):

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

class RNNLanguageModel:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.E = np.random.randn(vocab_size, embed_size) * 0.01
        self.W_xh = np.random.randn(hidden_size, embed_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((vocab_size, 1))

    def forward(self, input_indices):
        h = np.zeros((self.W_hh.shape[0], 1))
        outputs = []
        for idx in input_indices:
            x_t = self.E[idx].reshape(-1, 1)
            h = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            p = softmax(y)
            outputs.append(p)
        return outputs

# Example usage:
vocab = {'I':0, 'love':1, 'AI':2}
model = RNNLanguageModel(vocab_size=3, embed_size=4, hidden_size=5)
input_indices = [vocab['I'], vocab['love']]
outputs = model.forward(input_indices)
for t, p in enumerate(outputs):
    print(f"Time step {t+1} prediction:", p.ravel())
```

**Explanation:**
- Each word is mapped to an embedding vector.
- The RNN processes the sequence and predicts the next word's probability distribution.
- The output is a probability vector over the vocabulary at each time step.

---

Next: [Image Captioning](05_Image_Captioning.md) 