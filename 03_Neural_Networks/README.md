# Neural Networks and Backpropagation

This module explores the fundamental concepts of neural networks, focusing on Multi-layer Perceptrons (MLPs) and the backpropagation algorithm that enables their training.

## Multi-layer Perceptron (MLP)

A Multi-layer Perceptron is a feedforward artificial neural network that consists of multiple layers of neurons, where each neuron in one layer is connected to every neuron in the next layer.

### Architecture

An MLP with $L$ layers has the following structure:

```math
\text{Input Layer: } x \in \mathbb{R}^{d_0}
```

```math
\text{Hidden Layers: } h^{(l)} \in \mathbb{R}^{d_l}, \quad l = 1, 2, \ldots, L-1
```

```math
\text{Output Layer: } y \in \mathbb{R}^{d_L}
```

where $d_l$ is the number of neurons in layer $l$.

### Forward Propagation

The forward pass computes the output through each layer:

```math
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
```

where:
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ is the weight matrix for layer $l$
- $b^{(l)} \in \mathbb{R}^{d_l}$ is the bias vector for layer $l$
- $\sigma$ is the activation function
- $h^{(0)} = x$ (input layer)

### Activation Functions

#### Sigmoid Function

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

**Properties:**
- Range: $(0, 1)$
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- **Vanishing Gradient Problem**: Derivative approaches zero for large inputs

#### Tanh Function

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

**Properties:**
- Range: $(-1, 1)$
- Derivative: $\tanh'(x) = 1 - \tanh^2(x)$
- Zero-centered, but still suffers from vanishing gradients

#### ReLU (Rectified Linear Unit)

```math
\text{ReLU}(x) = \max(0, x)
```

**Properties:**
- Range: $[0, \infty)$
- Derivative: $\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$
- **Advantages**: Simple, fast, helps with vanishing gradient problem
- **Disadvantages**: Dying ReLU problem (neurons can become inactive)

#### Leaky ReLU

```math
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}
```

where $\alpha$ is a small positive constant (typically 0.01).

### Loss Function

For classification tasks, the output layer typically uses softmax activation:

```math
y_i = \frac{e^{z_i}}{\sum_{j=1}^{d_L} e^{z_j}}
```

where $z_i$ is the pre-activation output of the $i$-th neuron.

The cross-entropy loss is:

```math
L = -\sum_{i=1}^{d_L} t_i \log(y_i)
```

where $t_i$ is the target value for class $i$.

## Backpropagation

Backpropagation is an efficient algorithm for computing gradients of the loss function with respect to all parameters in the network.

### Chain Rule Foundation

Backpropagation is based on the chain rule of calculus:

```math
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial h^{(l)}} \frac{\partial h^{(l)}}{\partial W^{(l)}}
```

```math
\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial h^{(l)}} \frac{\partial h^{(l)}}{\partial b^{(l)}}
```

### Error Signal (Delta)

Define the error signal for layer $l$:

```math
\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}
```

where $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ is the pre-activation.

### Backpropagation Algorithm

#### Step 1: Forward Pass

Compute activations for all layers:

```math
z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}
```

```math
h^{(l)} = \sigma(z^{(l)})
```

#### Step 2: Initialize Output Layer Error

For the output layer $L$:

```math
\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial h^{(L)}} \odot \sigma'(z^{(L)})
```

For cross-entropy loss with softmax:

```math
\delta^{(L)} = h^{(L)} - t
```

where $t$ is the target vector.

#### Step 3: Backpropagate Error

For layers $l = L-1, L-2, \ldots, 1$:

```math
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})
```

#### Step 4: Compute Gradients

For each layer $l$:

```math
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T
```

```math
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
```

### Matrix Formulation

For mini-batch processing with batch size $B$:

```math
\frac{\partial L}{\partial W^{(l)}} = \frac{1}{B} \sum_{i=1}^{B} \delta_i^{(l)} (h_i^{(l-1)})^T
```

```math
\frac{\partial L}{\partial b^{(l)}} = \frac{1}{B} \sum_{i=1}^{B} \delta_i^{(l)}
```

### Computational Complexity

- **Forward Pass**: $O(\sum_{l=1}^{L} d_l d_{l-1})$
- **Backward Pass**: $O(\sum_{l=1}^{L} d_l d_{l-1})$
- **Memory**: $O(\sum_{l=1}^{L} d_l)$ for storing activations

### Numerical Stability

#### Gradient Clipping

Prevent exploding gradients:

```math
\text{if } \|\nabla L\| > \tau: \quad \nabla L \leftarrow \frac{\tau}{\|\nabla L\|} \nabla L
```

#### Batch Normalization

Normalize activations within each mini-batch:

```math
\mu_B = \frac{1}{B} \sum_{i=1}^{B} x_i
```

```math
\sigma_B^2 = \frac{1}{B} \sum_{i=1}^{B} (x_i - \mu_B)^2
```

```math
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
```

```math
y_i = \gamma \hat{x}_i + \beta
```

where $\gamma$ and $\beta$ are learnable parameters.

## Training Process

### Parameter Updates

Using gradient descent:

```math
W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}
```

```math
b^{(l)} \leftarrow b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}
```

where $\alpha$ is the learning rate.

### Mini-batch Training

1. **Forward Pass**: Compute predictions for mini-batch
2. **Compute Loss**: Calculate loss and gradients
3. **Backward Pass**: Backpropagate error signals
4. **Update Parameters**: Apply gradient updates
5. **Repeat**: Continue until convergence

### Convergence Criteria

- **Loss Threshold**: $L < \epsilon$
- **Gradient Norm**: $\|\nabla L\| < \epsilon$
- **Validation Performance**: Monitor validation loss/accuracy
- **Maximum Iterations**: Stop after $T$ epochs

## Implementation Considerations

### Weight Initialization

#### Xavier/Glorot Initialization

```math
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{in} + d_{out}}\right)
```

#### He Initialization (for ReLU)

```math
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{in}}\right)
```

### Regularization

#### L2 Regularization

```math
L_{total} = L + \frac{\lambda}{2} \sum_{l=1}^{L} \|W^{(l)}\|_F^2
```

#### Dropout

During training, randomly set activations to zero:

```math
h^{(l)} = \sigma(z^{(l)}) \odot m^{(l)}
```

where $m^{(l)} \sim \text{Bernoulli}(p)$.

### Hyperparameter Tuning

**Architecture:**
- Number of layers: 2-5 for most problems
- Hidden layer sizes: Start with $d_l = \sqrt{d_{l-1} d_{l+1}}$
- Activation functions: ReLU for hidden layers, softmax for output

**Training:**
- Learning rate: 0.001-0.1 (use learning rate scheduling)
- Batch size: 32-256
- Regularization: $\lambda = 0.0001-0.01$

## Advanced Topics

### Skip Connections (ResNet-style)

```math
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)}) + h^{(l-2)}
```

### Attention Mechanisms

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Modern Architectures

- **ResNet**: Skip connections for deep networks
- **DenseNet**: Dense connections between layers
- **Transformer**: Self-attention mechanisms

## Summary

Neural Networks and Backpropagation provide the foundation for modern deep learning:

1. **MLPs** offer universal approximation capabilities
2. **Backpropagation** enables efficient gradient computation
3. **Activation Functions** introduce non-linearity
4. **Regularization** prevents overfitting
5. **Optimization** techniques improve training

Understanding these concepts is essential for building and training effective neural networks. The combination of forward and backward passes allows neural networks to learn complex patterns from data through gradient-based optimization. 