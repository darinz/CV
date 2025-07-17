# Attention and Transformers

This module explores the revolutionary attention mechanism and Transformer architecture that have transformed natural language processing and computer vision.

## Self-Attention

Self-attention allows a model to attend to different parts of the input sequence when processing each element, capturing long-range dependencies effectively.

### Mathematical Formulation

For an input sequence $X = [x_1, x_2, \ldots, x_n]$, where $x_i \in \mathbb{R}^{d_{model}}$:

#### Query, Key, Value Projections

```math
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
```

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}$ are learnable weight matrices.

#### Attention Scores

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

The attention score between positions $i$ and $j$ is:

```math
\alpha_{ij} = \frac{\exp\left(\frac{q_i \cdot k_j^T}{\sqrt{d_k}}\right)}{\sum_{l=1}^{n} \exp\left(\frac{q_i \cdot k_l^T}{\sqrt{d_k}}\right)}
```

#### Output Computation

```math
\text{output}_i = \sum_{j=1}^{n} \alpha_{ij} v_j
```

### Scaled Dot-Product Attention

The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing too large:

```math
\text{ScaledAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Multi-Head Attention

Multi-head attention allows the model to attend to information from different representation subspaces:

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O
```

where each head is:

```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

and $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$, $W^O \in \mathbb{R}^{hd_k \times d_{model}}$.

### Attention Weights Visualization

The attention weights $\alpha_{ij}$ form a matrix $A \in \mathbb{R}^{n \times n}$:

```math
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
```

This matrix shows how much each position attends to every other position.

## Transformers

The Transformer architecture, introduced in "Attention Is All You Need," uses self-attention mechanisms to process sequences without recurrence.

### Architecture Overview

The Transformer consists of an encoder and decoder, each containing multiple identical layers.

#### Encoder Layer

```math
\text{EncoderLayer}(x) = \text{LayerNorm}(\text{FFN}(\text{LayerNorm}(x + \text{MultiHead}(x))) + \text{MultiHead}(x))
```

#### Decoder Layer

```math
\text{DecoderLayer}(x, y) = \text{LayerNorm}(\text{FFN}(\text{LayerNorm}(y + \text{MultiHead}(y, x))) + \text{MultiHead}(y, x))
```

### Positional Encoding

Since Transformers have no recurrence, positional information is injected through positional encodings:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

The input embedding becomes:

```math
X' = X + PE
```

### Feed-Forward Network

Each layer contains a feed-forward network:

```math
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
```

where $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$.

### Layer Normalization

Layer normalization normalizes across the feature dimension:

```math
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

where $\mu$ and $\sigma^2$ are computed across the feature dimension.

## Attention Variants

### Masked Self-Attention

For decoder layers, attention is masked to prevent attending to future positions:

```math
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
```

where $M_{ij} = -\infty$ if $i > j$, and $0$ otherwise.

### Relative Positional Encoding

Instead of absolute positions, use relative positions:

```math
\text{RelativeAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + R\right)V
```

where $R_{ij}$ encodes the relative position between $i$ and $j$.

### Sparse Attention

Reduce computational complexity by attending to a subset of positions:

```math
\text{SparseAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M\right)V
```

where $M$ is a sparse mask.

## Training and Optimization

### Loss Function

For sequence-to-sequence tasks:

```math
L = -\sum_{t=1}^{T} \log P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
```

### Learning Rate Scheduling

#### Warmup and Decay

```math
\text{lr}(t) = d_{model}^{-0.5} \cdot \min(t^{-0.5}, t \cdot \text{warmup\_steps}^{-1.5})
```

### Regularization

#### Dropout

```math
\text{Attention}(Q, K, V) = \text{dropout}(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right))V
```

#### Label Smoothing

```math
L = -\sum_{t=1}^{T} \sum_{c=1}^{C} (1 - \epsilon) \log P(y_t = c) + \frac{\epsilon}{C} \log P(y_t = c)
```

## Computational Complexity

### Time Complexity

For sequence length $n$ and model dimension $d$:

```math
\text{Time Complexity} = O(n^2 \cdot d)
```

### Space Complexity

```math
\text{Space Complexity} = O(n^2)
```

### Memory Usage

The attention matrix requires:

```math
\text{Memory} = O(n^2 \cdot h \cdot d_k)
```

where $h$ is the number of attention heads.

## Advanced Transformer Variants

### Transformer-XL

Addresses the fixed-length context limitation:

```math
h_t^{(l)} = \text{Attention}(Q_t^{(l)}, K_{t-L:t}^{(l)}, V_{t-L:t}^{(l)})
```

where $L$ is the segment length.

### Reformer

Efficient attention for long sequences:

```math
\text{LSHAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

with locality-sensitive hashing to reduce complexity.

### Performer

Linear attention mechanism:

```math
\text{LinearAttention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)\phi(K)^T}
```

where $\phi$ is a feature map.

## Applications

### Machine Translation

```math
P(y | x) = \prod_{t=1}^{T_y} P(y_t | y_1, y_2, \ldots, y_{t-1}, x)
```

### Language Modeling

```math
P(w_t | w_1, w_2, \ldots, w_{t-1}) = \text{softmax}(W_{out} h_t + b_{out})
```

### Text Classification

```math
y = \text{softmax}(W_{out} \text{mean}(H) + b_{out})
```

where $H$ is the sequence of hidden states.

### Named Entity Recognition

```math
P(t_i | w_1, w_2, \ldots, w_n) = \text{softmax}(W_{out} h_i + b_{out})
```

## Vision Transformers (ViT)

Transformers adapted for computer vision tasks.

### Patch Embedding

Split image into patches and embed:

```math
x_i = \text{Linear}(\text{Flatten}(P_i))
```

where $P_i$ is the $i$-th patch.

### Position Embedding

Add learnable position embeddings:

```math
z_0 = [x_{class}; x_1, x_2, \ldots, x_n] + E_{pos}
```

### Classification Head

```math
y = \text{MLP}(\text{LayerNorm}(z_L^0))
```

where $z_L^0$ is the class token at the final layer.

## Evaluation Metrics

### Perplexity

```math
\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(w_t | w_1, w_2, \ldots, w_{t-1})\right)
```

### BLEU Score

```math
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
```

### Accuracy

```math
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
```

## Implementation Considerations

### Memory Optimization

#### Gradient Checkpointing

```math
\text{Memory} = O(\sqrt{n^2 \cdot d})
```

#### Mixed Precision

Use FP16 for training to reduce memory usage.

### Parallelization

#### Model Parallelism

Distribute layers across multiple devices.

#### Data Parallelism

Process different batches on different devices.

### Hyperparameter Tuning

#### Model Size

- $d_{model}$: 512-2048
- $d_{ff}$: 2048-8192
- $h$: 8-16 heads
- $L$: 6-24 layers

#### Training

- Learning rate: $10^{-4}$ to $10^{-3}$
- Batch size: 32-4096
- Warmup steps: 4000-8000

## Summary

Attention and Transformers have revolutionized deep learning:

1. **Self-Attention**: Captures long-range dependencies efficiently
2. **Multi-Head Attention**: Attends to different representation subspaces
3. **Transformer Architecture**: Parallel processing without recurrence
4. **Positional Encoding**: Injects positional information
5. **Advanced Variants**: Address computational and memory limitations

Transformers have become the foundation for state-of-the-art models in NLP, computer vision, and other domains, enabling unprecedented performance on various tasks. 