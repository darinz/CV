# Training Techniques for RNNs

Training RNNs can be challenging due to issues like vanishing/exploding gradients and overfitting. Several techniques help stabilize and improve training.

## Gradient Clipping

Gradient clipping prevents exploding gradients by scaling down gradients that exceed a threshold $`\tau`$.

```math
\text{if } \|\nabla L\| > \tau: \quad \nabla L \leftarrow \frac{\tau}{\|\nabla L\|} \nabla L
```

**Python Example:**
```python
def clip_gradients(grads, tau):
    norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    if norm > tau:
        grads = [g * (tau / norm) for g in grads]
    return grads

# Example usage:
grads = [np.random.randn(3, 3), np.random.randn(3, 3)]
clipped = clip_gradients(grads, tau=1.0)
print('Clipped gradients:', clipped)
```

## Dropout

Dropout randomly sets a fraction $`p`$ of the hidden units to zero during training, helping prevent overfitting.

```math
h_t = \text{dropout}(h_t, p)
```

**Python Example:**
```python
def dropout(x, p):
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1 - p)

# Example usage:
h = np.random.randn(4, 1)
h_drop = dropout(h, p=0.5)
print('After dropout:', h_drop.ravel())
```

## Layer Normalization

Layer normalization normalizes the activations of a layer for each data point, improving training stability.

```math
h_t = \text{LayerNorm}(h_t)
```

**Python Example:**
```python
def layer_norm(x, eps=1e-5):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + eps)

# Example usage:
h = np.random.randn(4, 1)
h_norm = layer_norm(h)
print('After layer norm:', h_norm.ravel())
```

---

Next: [Applications](09_Applications.md) 