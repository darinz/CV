# 08. Performance Comparison

Comparing CNN architectures involves evaluating both accuracy and computational efficiency. This guide covers key metrics and efficiency considerations.

## ImageNet Top-5 Error Rates

| Architecture | Top-5 Error | Parameters |
|--------------|-------------|------------|
| AlexNet      |15.3      | 60        |
| VGG-16       | 7.3      | 138M       |
| ResNet-50   | 50.3      | 256M      |
| ResNet-11   | 40.6      | 44.5M      |

## Computational Efficiency

### FLOPs (Floating Point Operations)

```math
\text{FLOPs} = \sum_{l} H_l \cdot W_l \cdot C_l \cdot k_l^2 \cdot C_[object Object]l-1
```

- $`H_l, W_l`$: Height and width of feature map at layer $`l`$
- $`C_l`$: Number of output channels at layer $`l`$
- $`k_l`$: Kernel size at layer $`l`$

### Memory Usage

```math
\text[object Object]Memory} = \sum_{l} H_l \cdot W_l \cdot C_l \cdot 4 \text{ bytes}
```

### Python Example: Model Analysis

```python
import torch
import torchvision.models as models
from thop import profile

# Load a pre-trained model
model = models.resnet50(pretrained=True)
input_tensor = torch.randn(1224

# Calculate FLOPs and parameters
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs:[object Object]flops:,}")
print(f"Parameters: {params:,})
```

## Summary
- Performance comparison involves both accuracy and efficiency metrics
- FLOPs and memory usage are important for practical deployment
- Modern architectures balance accuracy with computational efficiency

---

**Next:** [Implementation Considerations](9mentation_Considerations.md) 