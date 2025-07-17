# Edge Computing for Computer Vision

This guide explains how to deploy and optimize computer vision models on edge devices, focusing on architecture, model partitioning, compression, and hardware acceleration.

## 1. Edge Deployment Architecture

Edge computing brings computation closer to data sources (e.g., cameras, sensors), reducing latency and bandwidth usage.

### Model Partitioning

Split the model between edge and cloud to balance latency and resource usage.

**Cloud-Edge Split:**
```math
f_{total} = f_{edge} \circ f_{cloud}
```

**Latency Model:**
```math
T_{total} = T_{edge} + T_{network} + T_{cloud}
```

Where:
```math
T_{network} = \frac{\text{data\_size}}{\text{bandwidth}} + \text{propagation\_delay}
```

#### Python Example: Simulate Latency
```python
def total_latency(t_edge, data_size, bandwidth, prop_delay, t_cloud):
    t_network = data_size / bandwidth + prop_delay
    return t_edge + t_network + t_cloud
```

### Edge-Cloud Optimization

Find the optimal split to minimize total latency.

**Optimal Split:**
```math
\text{Split}^* = \arg\min_{\text{split}} T_{total}(\text{split})
```

## 2. Model Compression

Reduce model size and computation for edge deployment.

### Quantization

Reduces precision of weights/activations (e.g., FP32 → INT8).

**Uniform Quantization:**
```math
Q(x) = \text{round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \times (2^b - 1)\right)
```

#### Python Example: Simple Quantization
```python
def quantize(x, x_min, x_max, bits=8):
    scale = (2**bits - 1) / (x_max - x_min)
    return np.round((x - x_min) * scale)
```

### Pruning

Removes unimportant weights or channels to create sparse models.

**Magnitude-based Pruning:**
```math
\text{Keep if } |w_{ij}| > \tau_{prune}
```

#### Python Example: Prune Small Weights
```python
def prune_weights(weights, threshold):
    pruned = np.where(np.abs(weights) > threshold, weights, 0)
    return pruned
```

### Knowledge Distillation

Train a small (student) model to mimic a large (teacher) model.

**Loss Function:**
```math
L_{KD} = \alpha L_{CE}(y, \hat{y}_s) + (1-\alpha) L_{KL}(T_s, T_t)
```

#### Python Example: Distillation Loss (PyTorch)
```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, T=2.0):
    ce_loss = F.cross_entropy(student_logits, targets)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * ce_loss + (1 - alpha) * kl_loss
```

## 3. Edge Hardware Optimization

### GPU Acceleration

Use CUDA or other accelerators for faster inference.

**Speedup:**
```math
\text{Speedup} = \frac{T_{CPU}}{T_{GPU}}
```

#### Python Example: Timing CPU vs GPU (PyTorch)
```python
import torch, time

def measure_speed(model, input_tensor):
    # CPU
    start = time.time()
    _ = model(input_tensor.cpu())
    cpu_time = time.time() - start
    # GPU
    if torch.cuda.is_available():
        start = time.time()
        _ = model(input_tensor.cuda())
        gpu_time = time.time() - start
    else:
        gpu_time = None
    return cpu_time, gpu_time
```

### Tensor Cores & Mixed Precision

Use FP16 for faster, lower-memory inference.

**Memory Reduction:**
```math
\text{FP16: } 16 \text{ bits per parameter}
\text{Memory Reduction: } \frac{1}{2} \text{ compared to FP32}
```

#### Python Example: Mixed Precision Inference
```python
import torch
from torch.cuda.amp import autocast

def infer_mixed_precision(model, input_tensor):
    with autocast():
        return model(input_tensor)
```

## Summary
- Edge computing reduces latency and bandwidth by processing data locally.
- Model compression (quantization, pruning, distillation) enables efficient edge deployment.
- Hardware acceleration (GPU, mixed precision) further optimizes performance. 