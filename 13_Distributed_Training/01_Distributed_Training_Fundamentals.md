# Distributed Training Fundamentals

Distributed training enables the training of large neural networks by distributing computation and memory across multiple devices or machines. This is essential for scaling up deep learning models that exceed the capacity of a single device.

## Parallelization Strategies

### Data Parallelism

**Concept:**
- Each worker (GPU or machine) gets a different subset of the data.
- All workers have a copy of the model.
- Each worker computes gradients on its data, then gradients are averaged (synchronized) and model parameters are updated.

**Math:**
If $N$ is the total number of samples and $K$ is the number of workers:

$$
\text{Each worker processes } \frac{N}{K} \text{ samples}
$$

Gradient synchronization:
$$
\nabla L = \frac{1}{K} \sum_{k=1}^K \nabla L_k
$$

**Python Example (PyTorch):**
```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
# (In practice, use torchrun or torch.distributed.launch)
dist.init_process_group("nccl")

model = nn.Linear(10, 2).cuda()
model = DDP(model)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward, backward, and optimizer step as usual
outputs = model(torch.randn(32, 10).cuda())
loss = outputs.sum()
loss.backward()
optimizer.step()
```

**Communication Cost:**
$$
T_{comm} = \frac{2 \times \text{model\_size}}{\text{bandwidth}}
$$

### Model Parallelism

**Concept:**
- The model is split across multiple devices.
- Each device holds a part of the model and processes a portion of the forward and backward pass.

**Layer-wise Partitioning:**
$$
\text{Layer } l \text{ on device } d = l \bmod D
$$
where $D$ is the number of devices.

**Pipeline Parallelism:**
- The model is split into stages, each on a different device.
- Micro-batches are pipelined through the stages.

**Python Example (PyTorch):**
```python
# Simple model parallelism example
import torch.nn as nn
import torch

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

class ModelParallelNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Linear(10, 50).to(device0)
        self.seq2 = nn.Linear(50, 2).to(device1)
    def forward(self, x):
        x = x.to(device0)
        x = torch.relu(self.seq1(x))
        x = x.to(device1)
        x = self.seq2(x)
        return x

model = ModelParallelNN()
output = model(torch.randn(32, 10))
```

### Hybrid Parallelism

**Concept:**
- Combines data, model, and pipeline parallelism for large-scale training.
- Used in training very large models (e.g., GPT-3, Megatron-LM).

**Math:**
$$
\text{DP groups} = \frac{\text{total\_devices}}{\text{tensor\_parallel\_size} \times \text{pipeline\_parallel\_size}}
$$

**Summary Table:**
| Strategy         | Model Copy | Data Split | Model Split | Use Case                |
|------------------|-----------|------------|-------------|-------------------------|
| Data Parallel    | Yes       | Yes        | No          | Most common, easy scale |
| Model Parallel   | No        | No         | Yes         | Very large models       |
| Pipeline Parallel| Partial   | Yes        | Yes         | Large, deep models      |
| Hybrid           | Partial   | Yes        | Yes         | Billion+ param models   |

---

This guide introduces the core strategies for distributed training. The next sections will cover synchronization, communication, memory, and advanced techniques in detail. 