# Mobile and Embedded Deployment

This guide explains how to deploy computer vision models on mobile and embedded devices, focusing on efficient architectures, optimization, and deployment strategies.

## 1. Mobile Neural Networks

### MobileNet Architecture
Uses depthwise separable convolutions for efficiency.

**Parameters:**
```math
\text{Parameters} = C \times K \times K + C \times C_{out}
```

**FLOPs:**
```math
\text{FLOPs} = H \times W \times C \times (K \times K + C_{out})
```

#### Python Example: MobileNet Block (PyTorch)
```python
import torch.nn as nn

class MobileNetBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, k, padding=k//2, groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

### ShuffleNet
Uses group convolutions and channel shuffle for speed.

**Channel Shuffle:**
```math
\text{Shuffle}(x) = \text{Reshape}(\text{Transpose}(\text{Reshape}(x)))
```

#### Python Example: Channel Shuffle
```python
import torch

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
```

## 2. Model Optimization for Mobile

### Mobile-Specific Pruning
Removes less important channels/filters.

#### Python Example: Channel Pruning
```python
import numpy as np

def prune_channels(weights, threshold):
    norms = np.linalg.norm(weights, ord=1, axis=(1,2,3))
    mask = norms > threshold
    return weights[mask]
```

### Mobile Quantization
Reduces memory and computation.

**INT8 Quantization:**
```math
\text{Memory Reduction: } \frac{1}{4} \text{ compared to FP32}
```

#### Python Example: Quantize to INT8
```python
import numpy as np

def quantize_int8(x):
    x = np.clip(x, -128, 127)
    return x.astype(np.int8)
```

## 3. Embedded System Optimization

### ARM NEON Optimization
Uses SIMD for fast vector operations.

**Vector Operations:**
```math
\text{Vector Operations: } 4 \text{ operations per cycle}
```

### DSP Optimization
Uses fixed-point arithmetic and LUTs.

**Q-format:**
```math
\text{Q-format: } Q_{m.n} \text{ where } m+n = \text{word\_length}
```

## 4. Deployment Strategies

### Model Conversion
Convert models to mobile/embedded formats.

#### TensorFlow Lite
```math
\text{Model Size} = \text{Original Size} \times \text{compression\_ratio}
```

#### Python Example: Convert to TFLite
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('model_dir')
tflite_model = converter.convert()
```

### Runtime Optimization
Fuse operations and use parallel execution.

#### Python Example: Operator Fusion (ONNX)
```python
import onnx
from onnx import optimizer
model = onnx.load('model.onnx')
passes = ["fuse_bn_into_conv"]
optimized = optimizer.optimize(model, passes)
```

## Summary
- Use efficient architectures (MobileNet, ShuffleNet) for mobile/embedded.
- Apply pruning and quantization for resource constraints.
- Optimize for hardware (NEON, DSP) and convert models for deployment. 