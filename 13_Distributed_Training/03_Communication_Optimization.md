# Communication Optimization in Distributed Training

Communication between devices can be a bottleneck in distributed training. Optimizing communication is crucial for scaling.

## Gradient Compression

### Quantization

**Concept:**
- Reduce the number of bits used to represent gradients (e.g., 1-bit SGD).

**Math:**
$$
\text{sign}(g) = \begin{cases} +1 & \text{if } g > 0 \\ -1 & \text{if } g \leq 0 \end{cases}
$$

**Python Example:**
```python
def quantize_gradients(grad):
    return torch.sign(grad)
```

### Sparsification

**Concept:**
- Only communicate the largest (top-k) gradients.
- Reduces communication volume.

**Python Example:**
```python
def topk_sparsify(grad, k):
    values, indices = torch.topk(grad.abs().flatten(), k)
    mask = torch.zeros_like(grad).flatten()
    mask[indices] = 1
    return (grad.flatten() * mask).reshape(grad.shape)
```

## Communication Scheduling

### Gradient Bucketing

**Concept:**
- Group gradients into buckets and communicate when a bucket is full.
- Overlaps communication with computation.

**Python Example (PyTorch):**
```python
# PyTorch DDP does this automatically, but you can set bucket size:
model = DDP(model, bucket_cap_mb=25)
```

---

Optimizing communication through compression and scheduling is key to efficient distributed training, especially at scale. 