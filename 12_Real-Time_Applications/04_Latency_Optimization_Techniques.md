# Latency Optimization Techniques

This guide covers methods to reduce inference time in real-time computer vision, including model, inference, and memory optimizations, with math and Python code.

## 1. Model Architecture Optimization

### Efficient Convolutions

#### Grouped Convolution
Splits input channels into groups, reducing computation.

**FLOPs:**
```math
\text{FLOPs} = \frac{H \times W \times C \times K \times K}{G}
```

#### Depthwise Separable Convolution
Performs spatial and channel mixing separately for efficiency.

**FLOPs:**
```math
\text{FLOPs} = H \times W \times (C + K \times K)
```

#### Python Example: Grouped Conv (PyTorch)
```python
import torch.nn as nn

grouped_conv = nn.Conv2d(32, 64, kernel_size=3, groups=4)
```

### Neural Architecture Search (NAS)
Automates model design under latency constraints.

**Search Space:**
```math
\mathcal{A} = \{\text{operations}, \text{connections}, \text{hyperparameters}\}
```

**Latency Constraint:**
```math
\text{Latency}(\mathcal{A}) \leq T_{target}
```

## 2. Inference Optimization

### Model Pruning
Removes unnecessary weights or channels.

#### Structured Pruning
```math
\text{Remove entire channels if } \|\mathbf{w}_c\|_2 < \tau
```

#### Unstructured Pruning
```math
\text{Remove individual weights if } |w_{ij}| < \tau
```

#### Python Example: Structured Pruning (PyTorch)
```python
import torch.nn.utils.prune as prune
prune.ln_structured(layer, name='weight', amount=0.2, n=2, dim=0)  # Prune 20% of channels
```

### Quantization
Reduces precision for faster inference.

#### Dynamic Quantization
```math
\text{Scale} = \frac{\text{max\_abs}}{\text{quant\_max}}
```

#### Static Quantization
Calibrate on a representative dataset.

#### Python Example: Dynamic Quantization (PyTorch)
```python
import torch
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

## 3. Memory Optimization

### Memory Pooling
Reuse memory for intermediate tensors.

**Peak Memory:**
```math
\text{Peak Memory} = \max_{l} \text{Memory}(l)
```

### Gradient Checkpointing
Saves memory by recomputing some activations during backward pass.

**Memory:**
```math
\text{Memory} = O(\sqrt{L}) \text{ instead of } O(L)
```

#### Python Example: Gradient Checkpointing (PyTorch)
```python
import torch.utils.checkpoint as cp
output = cp.checkpoint(model, input_tensor)
```

### Batch Processing
Finds optimal batch size for throughput.

**Optimal Batch Size:**
```math
\text{Batch}^* = \arg\max_{b} \frac{\text{throughput}(b)}{b}
```

## Summary
- Optimize model architecture (grouped/depthwise conv, NAS) for speed.
- Use pruning and quantization for faster inference.
- Apply memory pooling, checkpointing, and batch tuning for memory efficiency. 