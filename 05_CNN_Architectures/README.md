# CNN Architectures

This module explores advanced CNN architectures and training techniques, focusing on Batch Normalization, Transfer Learning, and the evolution of architectures from AlexNet to modern ResNet variants.

## Batch Normalization

Batch Normalization (BatchNorm) is a technique that normalizes the inputs of each layer, significantly improving training speed and stability of deep neural networks.

### Mathematical Formulation

For a mini-batch $\mathcal{B} = \{x_1, x_2, \ldots, x_m\}$:

#### Training Phase

**Compute mini-batch statistics:**
```math
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i
```

```math
\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2
```

**Normalize:**
```math
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
```

**Scale and shift:**
```math
y_i = \gamma \hat{x}_i + \beta
```

where $\gamma$ and $\beta$ are learnable parameters, and $\epsilon$ is a small constant for numerical stability.

#### Inference Phase

During inference, use running averages:

```math
\mu_{running} = \alpha \mu_{running} + (1 - \alpha) \mu_\mathcal{B}
```

```math
\sigma_{running}^2 = \alpha \sigma_{running}^2 + (1 - \alpha) \sigma_\mathcal{B}^2
```

```math
y_i = \gamma \frac{x_i - \mu_{running}}{\sqrt{\sigma_{running}^2 + \epsilon}} + \beta
```

### Benefits

1. **Faster Training**: Allows higher learning rates
2. **Reduced Internal Covariate Shift**: Stabilizes layer inputs
3. **Regularization Effect**: Adds noise during training
4. **Reduced Dependence on Initialization**: More robust to weight initialization

### Gradient Computation

The gradients with respect to the parameters are:

```math
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \hat{x}_i
```

```math
\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}
```

```math
\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \left(\frac{\partial L}{\partial y_i} - \frac{1}{m} \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} - \frac{\hat{x}_i}{m} \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \hat{x}_j\right)
```

### Variants

#### Layer Normalization

Normalize across features for each sample:

```math
\mu_i = \frac{1}{H} \sum_{j=1}^{H} x_{i,j}
```

```math
\sigma_i^2 = \frac{1}{H} \sum_{j=1}^{H} (x_{i,j} - \mu_i)^2
```

#### Instance Normalization

Normalize across spatial dimensions for each sample and channel:

```math
\mu_{i,c} = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{i,h,w,c}
```

```math
\sigma_{i,c}^2 = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{i,h,w,c} - \mu_{i,c})^2
```

## Transfer Learning

Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks with limited data.

### Pre-training Phase

Train a model on a large dataset (e.g., ImageNet):

```math
\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} L(f_\theta(x_i), y_i)
```

### Fine-tuning Strategies

#### Feature Extraction

Freeze pre-trained layers and train only new layers:

```math
\theta_{frozen} = \theta_{pretrained}
```

```math
\theta_{new} = \text{random initialization}
```

#### Fine-tuning

Update all parameters with lower learning rate:

```math
\theta_{new} = \theta_{pretrained} - \alpha \nabla_\theta L(\theta_{pretrained})
```

where $\alpha$ is typically smaller than pre-training learning rate.

### Learning Rate Scheduling

#### Differential Learning Rates

```math
\alpha_l = \begin{cases}
\alpha_{base} \cdot 0.1 & \text{for frozen layers} \\
\alpha_{base} & \text{for new layers}
\end{cases}
```

#### Gradual Unfreezing

Progressively unfreeze layers:

```math
\text{Stage 1: } \theta_{trainable} = \{\theta_{new}\}
```

```math
\text{Stage 2: } \theta_{trainable} = \{\theta_{new}, \theta_{L-1}\}
```

```math
\text{Stage 3: } \theta_{trainable} = \{\theta_{new}, \theta_{L-1}, \theta_{L-2}\}
```

### Domain Adaptation

#### Maximum Mean Discrepancy (MMD)

```math
\text{MMD}(P, Q) = \left\|\mathbb{E}_{x \sim P}[\phi(x)] - \mathbb{E}_{y \sim Q}[\phi(y)]\right\|_{\mathcal{H}}^2
```

where $\phi$ is a feature mapping.

## AlexNet

AlexNet (2012) was the first deep CNN to win the ImageNet challenge, marking the beginning of the deep learning revolution.

### Architecture

**Input:** $227 \times 227 \times 3$

**Layer 1:** Conv(96, 11×11, stride=4) → ReLU → MaxPool(3×3, stride=2)
```math
\text{Output: } 55 \times 55 \times 96
```

**Layer 2:** Conv(256, 5×5, pad=2) → ReLU → MaxPool(3×3, stride=2)
```math
\text{Output: } 27 \times 27 \times 256
```

**Layer 3:** Conv(384, 3×3, pad=1) → ReLU
```math
\text{Output: } 27 \times 27 \times 384
```

**Layer 4:** Conv(384, 3×3, pad=1) → ReLU
```math
\text{Output: } 27 \times 27 \times 384
```

**Layer 5:** Conv(256, 3×3, pad=1) → ReLU → MaxPool(3×3, stride=2)
```math
\text{Output: } 13 \times 13 \times 256
```

**Layer 6:** FC(4096) → ReLU → Dropout(0.5)
```math
\text{Output: } 4096
```

**Layer 7:** FC(4096) → ReLU → Dropout(0.5)
```math
\text{Output: } 4096
```

**Layer 8:** FC(1000) → Softmax
```math
\text{Output: } 1000
```

### Key Innovations

1. **ReLU Activation**: Faster training than sigmoid/tanh
2. **Dropout**: Regularization to prevent overfitting
3. **Data Augmentation**: Random crops, horizontal flips
4. **GPU Training**: First large-scale GPU implementation

### Parameters

```math
\text{Total Parameters} = \sum_{l} (k_l^2 \cdot c_{l-1} \cdot c_l + c_l)
```

```math
\text{Parameters} \approx 60 \text{ million}
```

## VGG

VGG (2014) introduced the concept of using small 3×3 convolutions in deep networks, achieving excellent performance through depth.

### Architecture Design Principles

1. **Small Kernels**: 3×3 convolutions with padding=1
2. **Increasing Depth**: More layers for better feature learning
3. **Doubling Channels**: Channels double after each pooling layer
4. **MaxPooling**: 2×2 max pooling for spatial reduction

### VGG-16 Architecture

**Input:** $224 \times 224 \times 3$

**Block 1:** Conv(64, 3×3) → ReLU → Conv(64, 3×3) → ReLU → MaxPool(2×2)
```math
\text{Output: } 112 \times 112 \times 64
```

**Block 2:** Conv(128, 3×3) → ReLU → Conv(128, 3×3) → ReLU → MaxPool(2×2)
```math
\text{Output: } 56 \times 56 \times 128
```

**Block 3:** Conv(256, 3×3) → ReLU → Conv(256, 3×3) → ReLU → Conv(256, 3×3) → ReLU → MaxPool(2×2)
```math
\text{Output: } 28 \times 28 \times 256
```

**Block 4:** Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → MaxPool(2×2)
```math
\text{Output: } 14 \times 14 \times 512
```

**Block 5:** Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → Conv(512, 3×3) → ReLU → MaxPool(2×2)
```math
\text{Output: } 7 \times 7 \times 512
```

**FC Layers:** FC(4096) → ReLU → Dropout(0.5) → FC(4096) → ReLU → Dropout(0.5) → FC(1000) → Softmax

### Mathematical Analysis

#### Receptive Field

For 3×3 convolutions with padding=1:
```math
\text{RF}_l = \text{RF}_{l-1} + 2
```

#### Parameters per Layer

```math
\text{Params} = k^2 \cdot c_{in} \cdot c_{out} + c_{out}
```

For 3×3 convolution:
```math
\text{Params} = 9 \cdot c_{in} \cdot c_{out} + c_{out}
```

### VGG Variants

- **VGG-11**: 8 conv + 3 FC layers
- **VGG-13**: 10 conv + 3 FC layers
- **VGG-16**: 13 conv + 3 FC layers
- **VGG-19**: 16 conv + 3 FC layers

## ResNet

ResNet (2015) introduced skip connections (residual connections) to enable training of very deep networks by addressing the vanishing gradient problem.

### Residual Block

The fundamental building block of ResNet:

```math
h^{(l+1)} = \mathcal{F}(h^{(l)}) + h^{(l)}
```

where $\mathcal{F}(h^{(l)})$ is the residual function.

#### Bottleneck Block

For deeper networks, use bottleneck design:

```math
h^{(l+1)} = h^{(l)} + W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot h^{(l)}))
```

where:
- $W_1 \in \mathbb{R}^{64 \times 256}$ (1×1 conv)
- $W_2 \in \mathbb{R}^{64 \times 64}$ (3×3 conv)
- $W_3 \in \mathbb{R}^{256 \times 64}$ (1×1 conv)

### Mathematical Properties

#### Gradient Flow

The gradient can flow directly through skip connections:

```math
\frac{\partial L}{\partial h^{(l)}} = \frac{\partial L}{\partial h^{(l+1)}} \cdot \frac{\partial h^{(l+1)}}{\partial h^{(l)}} = \frac{\partial L}{\partial h^{(l+1)}} \cdot (1 + \frac{\partial \mathcal{F}}{\partial h^{(l)}})
```

#### Identity Mapping

When $\mathcal{F}(h^{(l)}) = 0$:
```math
h^{(l+1)} = h^{(l)}
```

This allows the network to learn identity mappings when optimal.

### ResNet Architectures

#### ResNet-50

**Stage 1:** Conv(64, 7×7, stride=2) → MaxPool(3×3, stride=2)
```math
\text{Output: } 56 \times 56 \times 64
```

**Stage 2:** 3 bottleneck blocks, channels: 64→256
```math
\text{Output: } 56 \times 56 \times 256
```

**Stage 3:** 4 bottleneck blocks, channels: 128→512
```math
\text{Output: } 28 \times 28 \times 512
```

**Stage 4:** 6 bottleneck blocks, channels: 256→1024
```math
\text{Output: } 14 \times 14 \times 1024
```

**Stage 5:** 3 bottleneck blocks, channels: 512→2048
```math
\text{Output: } 7 \times 7 \times 2048
```

**Output:** GlobalAvgPool → FC(1000) → Softmax

#### ResNet Variants

- **ResNet-18**: 18 layers, basic blocks
- **ResNet-34**: 34 layers, basic blocks
- **ResNet-50**: 50 layers, bottleneck blocks
- **ResNet-101**: 101 layers, bottleneck blocks
- **ResNet-152**: 152 layers, bottleneck blocks

### Advanced ResNet Variants

#### ResNeXt

Grouped convolutions for better efficiency:

```math
h^{(l+1)} = h^{(l)} + \sum_{g=1}^{G} \mathcal{F}_g(h^{(l)})
```

where $G$ is the number of groups.

#### DenseNet

Each layer connects to all subsequent layers:

```math
h^{(l)} = \mathcal{H}_l([h^{(0)}, h^{(1)}, \ldots, h^{(l-1)}])
```

where $[\cdot]$ denotes concatenation.

## Training Strategies

### Learning Rate Scheduling

#### Step Decay

```math
\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}
```

#### Cosine Annealing

```math
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}{T}\pi))
```

### Data Augmentation

#### Training Augmentations

```math
\text{Random Crop: } I' = \text{crop}(I, 224 \times 224)
```

```math
\text{Horizontal Flip: } I' = \text{flip}(I, p=0.5)
```

```math
\text{Color Jittering: } I' = I \odot \text{color\_transform}
```

#### Test Augmentations

```math
\text{Center Crop: } I' = \text{center\_crop}(I, 224 \times 224)
```

```math
\text{10-Crop: } I' = \text{ensemble}(\text{10 crops})
```

## Performance Comparison

### ImageNet Top-5 Error Rates

| Architecture | Top-5 Error | Parameters |
|--------------|-------------|------------|
| AlexNet      | 15.3%       | 60M        |
| VGG-16       | 7.3%        | 138M       |
| ResNet-50    | 5.3%        | 25.6M      |
| ResNet-101   | 4.6%        | 44.5M      |

### Computational Efficiency

```math
\text{FLOPs} = \sum_{l} H_l \cdot W_l \cdot C_l \cdot k_l^2 \cdot C_{l-1}
```

## Summary

CNN architectures have evolved significantly:

1. **AlexNet**: First deep CNN, introduced ReLU and dropout
2. **VGG**: Demonstrated effectiveness of depth with 3×3 convolutions
3. **ResNet**: Solved vanishing gradient problem with skip connections
4. **BatchNorm**: Improved training stability and speed
5. **Transfer Learning**: Enabled effective use of pre-trained models

These architectures and techniques form the foundation for modern computer vision systems, enabling state-of-the-art performance across various tasks. 