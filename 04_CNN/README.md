# Image Classification with CNNs

This module explores Convolutional Neural Networks (CNNs), the dominant architecture for image classification and computer vision tasks, covering their history, mathematical foundations, and key components.

## History

### Early Developments

The foundations of CNNs trace back to several key developments:

**1960s - Hubel and Wiesel**: Discovered that visual cortex neurons respond to specific patterns and orientations, inspiring the concept of receptive fields.

**1980s - Neocognitron**: Fukushima's hierarchical neural network with local receptive fields and shared weights.

**1989 - Backpropagation for CNNs**: LeCun's application of backpropagation to convolutional networks.

### Modern Era

**2012 - AlexNet**: Krizhevsky et al. achieved breakthrough performance on ImageNet, marking the beginning of the deep learning revolution in computer vision.

**2014 - VGGNet**: Simonyan and Zisserman introduced deeper networks with 3×3 convolutions.

**2015 - ResNet**: He et al. introduced skip connections, enabling training of very deep networks.

**2017 - Transformer**: Vaswani et al. introduced attention mechanisms, later adapted for vision tasks.

## Higher-level Representations and Image Features

### Hierarchical Feature Learning

CNNs learn hierarchical representations through multiple layers:

```math
\text{Low-level features: } \text{edges, corners, textures}
```

```math
\text{Mid-level features: } \text{shapes, patterns, parts}
```

```math
\text{High-level features: } \text{objects, scenes, semantics}
```

### Feature Hierarchy

Each layer extracts increasingly abstract features:

```math
\text{Layer 1: } f_1(x) = \text{edge detectors, color blobs}
```

```math
\text{Layer 2: } f_2(f_1(x)) = \text{combinations of edges, textures}
```

```math
\text{Layer 3: } f_3(f_2(f_1(x))) = \text{object parts, shapes}
```

```math
\text{Layer L: } f_L(\ldots) = \text{semantic concepts, object classes}
```

### Receptive Field

The receptive field of a neuron is the region of the input that affects its output:

```math
\text{RF}_l = \text{RF}_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i
```

where:
- $k_l$ is the kernel size at layer $l$
- $s_i$ is the stride at layer $i$

## Convolution

Convolution is the fundamental operation in CNNs, enabling parameter sharing and translation invariance.

### 2D Convolution

For an input image $I \in \mathbb{R}^{H \times W \times C}$ and kernel $K \in \mathbb{R}^{k \times k \times C}$:

```math
(I * K)_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{i+m, j+n, c} \cdot K_{m,n,c}
```

### Multiple Kernels

With $F$ kernels, the output becomes:

```math
O \in \mathbb{R}^{H' \times W' \times F}
```

where each output channel $f$ is:

```math
O_{i,j,f} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{i+m, j+n, c} \cdot K_{m,n,c,f}
```

### Stride

Stride controls the step size of the convolution:

```math
O_{i,j,f} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{s \cdot i+m, s \cdot j+n, c} \cdot K_{m,n,c,f}
```

where $s$ is the stride.

### Padding

Padding preserves spatial dimensions:

```math
H' = \frac{H - k + 2p}{s} + 1
```

```math
W' = \frac{W - k + 2p}{s} + 1
```

where $p$ is the padding size.

### Transposed Convolution (Deconvolution)

For upsampling:

```math
O_{i,j,f} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \sum_{c=0}^{C-1} I_{\lfloor i/s \rfloor + m, \lfloor j/s \rfloor + n, c} \cdot K_{m,n,c,f}
```

## Pooling

Pooling operations reduce spatial dimensions while preserving important features.

### Max Pooling

```math
\text{MaxPool}(I)_{i,j} = \max_{m,n \in \mathcal{R}_{i,j}} I_{m,n}
```

where $\mathcal{R}_{i,j}$ is the pooling region centered at $(i,j)$.

### Average Pooling

```math
\text{AvgPool}(I)_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{m,n \in \mathcal{R}_{i,j}} I_{m,n}
```

### Global Pooling

```math
\text{GlobalAvgPool}(I) = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} I_{i,j}
```

```math
\text{GlobalMaxPool}(I) = \max_{i,j} I_{i,j}
```

## CNN Architecture Components

### Convolutional Layer

```math
h^{(l)} = \sigma(W^{(l)} * h^{(l-1)} + b^{(l)})
```

where $*$ denotes convolution.

### Pooling Layer

```math
h^{(l)} = \text{Pool}(h^{(l-1)})
```

### Fully Connected Layer

```math
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
```

## Modern CNN Architectures

### VGGNet

Uses repeated blocks of 3×3 convolutions:

```math
\text{Block: } \text{Conv}(3×3, 64) \rightarrow \text{ReLU} \rightarrow \text{Conv}(3×3, 64) \rightarrow \text{ReLU} \rightarrow \text{MaxPool}(2×2)
```

### ResNet

Skip connections enable training of very deep networks:

```math
h^{(l)} = \mathcal{F}(h^{(l-1)}) + h^{(l-1)}
```

where $\mathcal{F}$ is the residual function.

### DenseNet

Each layer connects to all subsequent layers:

```math
h^{(l)} = \mathcal{H}_l([h^{(0)}, h^{(1)}, \ldots, h^{(l-1)}])
```

where $[\cdot]$ denotes concatenation.

## Training CNNs

### Loss Function

For classification with $C$ classes:

```math
L = -\sum_{i=1}^{C} t_i \log(y_i)
```

where $t_i$ is the target and $y_i$ is the predicted probability.

### Data Augmentation

Common augmentations:

```math
\text{Rotation: } I' = R_\theta(I)
```

```math
\text{Translation: } I'(x,y) = I(x + \Delta x, y + \Delta y)
```

```math
\text{Scaling: } I' = \text{resize}(I, \alpha \cdot \text{size}(I))
```

```math
\text{Color Jittering: } I' = I \odot \text{color\_transform}
```

### Regularization

#### Dropout

```math
h^{(l)} = h^{(l)} \odot m^{(l)}
```

where $m^{(l)} \sim \text{Bernoulli}(p)$.

#### Batch Normalization

```math
\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
```

## Feature Visualization

### Activation Maps

The activation of a specific filter:

```math
A_{i,j}^{(l,f)} = h_{i,j,f}^{(l)}
```

### Grad-CAM

Class activation mapping:

```math
\text{Grad-CAM} = \text{ReLU}\left(\sum_{f} \alpha_f A^{(l,f)}\right)
```

where:

```math
\alpha_f = \frac{1}{HW} \sum_{i,j} \frac{\partial y_c}{\partial A_{i,j}^{(l,f)}}
```

## Transfer Learning

### Fine-tuning

Update pre-trained weights:

```math
\theta_{new} = \theta_{pretrained} - \alpha \nabla_\theta L(\theta_{pretrained})
```

### Feature Extraction

Freeze pre-trained layers:

```math
\theta_{frozen} = \theta_{pretrained}
```

```math
\theta_{trainable} = \text{random initialization}
```

## Performance Metrics

### Top-1 Accuracy

```math
\text{Top-1 Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max(y_i) = t_i]
```

### Top-5 Accuracy

```math
\text{Top-5 Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[t_i \in \text{top5}(y_i)]
```

### Mean Average Precision (mAP)

```math
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
```

where $\text{AP}_c$ is the average precision for class $c$.

## Implementation Considerations

### Memory Efficiency

- **Gradient Checkpointing**: Trade computation for memory
- **Mixed Precision**: Use FP16 for training
- **Model Parallelism**: Distribute across multiple GPUs

### Computational Optimization

- **Winograd Algorithm**: Efficient convolution
- **FFT-based Convolution**: For large kernels
- **Depthwise Separable Convolution**: Reduce parameters

### Hyperparameter Tuning

**Architecture:**
- Kernel sizes: 3×3, 5×5, 7×7
- Number of filters: 32, 64, 128, 256, 512
- Pooling: Max, Average, Global

**Training:**
- Learning rate: 0.001-0.1
- Batch size: 16-256
- Optimizer: SGD with momentum, Adam

## Summary

Convolutional Neural Networks have revolutionized computer vision:

1. **History**: From biological inspiration to modern architectures
2. **Hierarchical Features**: Learning representations from low to high level
3. **Convolution**: Parameter sharing and translation invariance
4. **Pooling**: Dimensionality reduction and feature preservation
5. **Modern Architectures**: ResNet, DenseNet, and beyond

CNNs continue to be the foundation for most computer vision tasks, providing powerful tools for image classification, object detection, segmentation, and more. 