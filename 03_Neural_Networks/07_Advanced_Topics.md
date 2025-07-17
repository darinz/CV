# 07. Advanced Topics

Modern neural networks use advanced architectural features to improve learning, stability, and performance. Here are some key concepts.

## Skip Connections (ResNet-style)

Skip connections allow information to bypass one or more layers, helping train very deep networks:

```math
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)}) + h^{(l-2)}
```

- **Benefit:** Helps gradients flow backward, reduces vanishing gradient problem

### Python Example: Skip Connection

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Assume h_prev2, h_prev1, W, b are defined
h = relu(W @ h_prev1 + b) + h_prev2
```

## Attention Mechanisms

Attention allows the network to focus on relevant parts of the input:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

- $`Q`$: Query matrix
- $`K`$: Key matrix
- $`V`$: Value matrix
- $`d_k`$: Dimension of keys

### Python Example: Scaled Dot-Product Attention

```python
import numpy as np

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return weights @ V
```

## Modern Architectures

- **ResNet:** Uses skip connections for very deep networks
- **DenseNet:** Connects each layer to every other layer
- **Transformer:** Uses self-attention for sequence modeling

## Summary
- Skip connections and attention mechanisms are key to modern deep learning
- These features enable training of deeper, more powerful networks
- Understanding these concepts is essential for advanced neural network design

---

**End of Neural Networks and Backpropagation Module** 