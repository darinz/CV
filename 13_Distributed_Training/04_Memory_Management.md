# Memory Management in Distributed Training

Efficient memory management is crucial for training large models on limited hardware. Techniques like gradient checkpointing and mixed precision training help reduce memory usage.

## Gradient Checkpointing

### Concept
- Instead of storing all intermediate activations for backpropagation, only a subset (checkpoints) are saved.
- Other activations are recomputed during the backward pass, saving memory at the cost of extra computation.

**Math:**
$$
\text{Memory} = O(\sqrt{L}) \text{ instead of } O(L)
$$
where $L$ is the number of layers.

**Python Example (PyTorch):**
```python
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        def custom_forward(*inputs):
            return self.seq(*inputs)
        return checkpoint.checkpoint(custom_forward, x)
```

## Mixed Precision Training

### Concept
- Use FP16 (half-precision) instead of FP32 (single-precision) for activations and gradients.
- Reduces memory usage and can speed up training on modern GPUs.

**Math:**
$$
\text{Memory Reduction} = \frac{1}{2} \text{ for activations and gradients}
$$

**Python Example (PyTorch):**
```python
import torch.cuda.amp as amp

model = ...
optimizer = ...
scaler = amp.GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    with amp.autocast():
        output = model(data)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

Memory management techniques like checkpointing and mixed precision are essential for scaling deep learning to larger models and datasets. 