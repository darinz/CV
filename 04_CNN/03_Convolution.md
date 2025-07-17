# 03. Convolution Operations in CNNs

Convolution is the fundamental operation in CNNs, enabling parameter sharing and translation invariance. This guide explains the math and implementation of convolutions in deep learning.

## 2D Convolution

For an input image $`I \in \mathbb{R}^{H \times W \times C}`$ and kernel $`K \in \mathbb{R}^{k \times k \times C}`$:

```math
(I * K)_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{i+m, j+n, c} \cdot K_{m,n,c}
```

- $`H, W`$: Height and width of the input
- $`C`$: Number of channels
- $`k`$: Kernel size

## Multiple Kernels (Filters)

With $`F`$ kernels, the output becomes:

```math
O \in \mathbb{R}^{H' \times W' \times F}
```

where each output channel $`f`$ is:

```math
O_{i,j,f} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{i+m, j+n, c} \cdot K_{m,n,c,f}
```

## Stride

Stride controls the step size of the convolution:

```math
O_{i,j,f} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{s \cdot i+m, s \cdot j+n, c} \cdot K_{m,n,c,f}
```

where $`s`$ is the stride.

## Padding

Padding preserves spatial dimensions:

```math
H' = \frac{H - k + 2p}{s} + 1
```
```math
W' = \frac{W - k + 2p}{s} + 1
```

where $`p`$ is the padding size.

## Transposed Convolution (Deconvolution)

For upsampling:

```math
O_{i,j,f} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{\lfloor i/s \rfloor + m, \lfloor j/s \rfloor + n, c} \cdot K_{m,n,c,f}
```

## Python Example: 2D Convolution (PyTorch)

```python
import torch
import torch.nn as nn

# 1 input channel, 1 output channel, 3x3 kernel
conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

# Example input: batch of 1 image, 1 channel, 5x5 pixels
x = torch.randn(1, 1, 5, 5)
y = conv(x)
print(y.shape)  # Output: torch.Size([1, 1, 5, 5])
```

## Summary
- Convolution enables parameter sharing and translation invariance
- Stride and padding control output size and spatial coverage
- Multiple filters extract different features from the input

---

**Next:** [Pooling Operations](04_Pooling.md) 