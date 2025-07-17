# Self-Attention Mechanism

Self-attention is the core innovation that powers modern Transformer architectures. It allows a model to attend to different parts of the input sequence when processing each element, capturing long-range dependencies effectively.

## What is Self-Attention?

Self-attention is a mechanism that computes a weighted sum of all input elements, where the weights are learned based on the relationships between elements. Unlike RNNs that process sequences sequentially, self-attention can directly access any position in the sequence, making it highly parallelizable and effective at capturing long-range dependencies.

### Key Concepts
1*Query (Q)**: What were looking for
2. **Key (K)**: What we're matching against3*Value (V)**: What we actually retrieve4*Attention Weights**: How much to focus on each element

## Mathematical Foundation

### Input Representation

For an input sequence $`X = [x_1, x_2ots, x_n]`$, where each $`x_i \in \mathbb{R}^{d_{model}}`$:

```math
X = \begin{bmatrix} 
x_1^T \\
x_2T \\
\vdots \\
x_n^T
\end{bmatrix} \in \mathbb{R}^{n \times d_{model}}
```

### Query, Key, Value Projections

The input is projected into three different spaces using learnable weight matrices:

```math
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
```

where:
- $`W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_k}`$ are learnable weight matrices
- $`d_k`$ is the dimension of the key/query vectors (typically $`d_{model}/h`$ where $`h`$ is the number of heads)

### Attention Scores Computation

The attention scores measure the compatibility between queries and keys:

```math
\text{Scores} = \frac{QK^T}{\sqrt{d_k}}
```

The scaling factor $`\frac{1}{\sqrt{d_k}}`$ prevents the dot products from growing too large in magnitude, which would push the softmax function into regions with extremely small gradients.

### Attention Weights

The scores are passed through a softmax function to obtain attention weights:

```math
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
```

Each element $`A_{ij}`$ represents how much position $`i`$ should attend to position $`j`$:

```math
A_{ij} = \frac{\exp\left(\frac{q_i \cdot k_j^T}{\sqrt{d_k}}\right)}{\sum_{l=1n} \exp\left(\frac{q_i \cdot k_l^T}{\sqrt{d_k}}\right)}
```

### Output Computation

The final output is computed as a weighted sum of values:

```math
\text{Attention}(Q, K, V) = AV
```

For each position $`i`$:

```math
\text{output}_i = \sum_{j=1} A_{ij} v_j
```

## Python Implementation

Let's implement self-attention from scratch to understand the mechanism:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads=1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)
        self.W_o = nn.Linear(d_v * n_heads, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_v
        )
        output = self.W_o(context)
        
        return output, attention_weights

# Example usage
def demonstrate_self_attention():
    # Parameters
    batch_size = 2   seq_len = 5
    d_model = 64    d_k = 32
    d_v = 32
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize self-attention
    self_attention = SelfAttention(d_model, d_k, d_v)
    
    # Forward pass
    output, attention_weights = self_attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(fAttention weights shape: {attention_weights.shape})  # Show attention weights for first batch, first head
    print(f"\nAttention weights (batch0ad 0):")
    print(attention_weights[0,0detach().numpy())
    
    return output, attention_weights

# Run demonstration
if __name__ == "__main__":
    output, attention_weights = demonstrate_self_attention()
```

## Understanding the Attention Weights

The attention weights matrix $`A \in \mathbb{R}^{n \times n}`$ is crucial for understanding how the model processes information:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_weights(attention_weights, title=Attention Weights"):
    """Visualize attention weights as a heatmap.""# Take first batch, first head
    weights = attention_weights[0,0.detach().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights, annot=True, cmap='Blues', fmt='.3   plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel(Query Position')
    plt.show()

# Example with meaningful input
def attention_analysis_example():
    # Create input with clear patterns
    seq_len = 6d_model = 64
    
    # Create input where each position has a unique pattern
    x = torch.randn(1, seq_len, d_model)
    
    # Make positions 0 and 3 similar
    x0, 3 = x[0, 0] +00.1orch.randn(d_model)
    
    # Make positions 1 and 4 similar
    x0, 4 = x[0, 1] +00.1orch.randn(d_model)
    
    # Initialize self-attention
    self_attention = SelfAttention(d_model, d_model//2, d_model//2)
    
    # Forward pass
    output, attention_weights = self_attention(x)
    
    # Visualize attention weights
    visualize_attention_weights(attention_weights, Self-Attention Weights)  return attention_weights

# Run analysis
attention_analysis_example()
```

## Scaled Dot-Product Attention

The scaling factor $`\frac{1}{\sqrt{d_k}}`$ is crucial for stable training:

```python
def demonstrate_scaling_importance():
    strate why scaling is important in attention.   
    # Simulate attention scores without scaling
    d_k =64seq_len = 10
    
    # Random Q and K matrices
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    
    # Compute scores with and without scaling
    scores_unscaled = torch.matmul(Q, K.T)
    scores_scaled = torch.matmul(Q, K.T) / math.sqrt(d_k)
    
    # Apply softmax
    attention_unscaled = F.softmax(scores_unscaled, dim=-1)
    attention_scaled = F.softmax(scores_scaled, dim=-1)
    
    print(fUnscaled scores - Mean: {scores_unscaled.mean():0.3}, Std: {scores_unscaled.std():.3)
    print(f"Scaled scores - Mean: {scores_scaled.mean():0.3}, Std: {scores_scaled.std():.3f}")
    
    print(f"\nUnscaled attention - Entropy: {-(attention_unscaled * torch.log(attention_unscaled +1e-8)).sum(dim=-1.mean():.3)
    print(f"Scaled attention - Entropy: {-(attention_scaled * torch.log(attention_scaled +1e-8)).sum(dim=-1ean():.3f})  return attention_unscaled, attention_scaled

# Run scaling demonstration
unscaled, scaled = demonstrate_scaling_importance()
```

## Multi-Head Attention

Multi-head attention allows the model to attend to information from different representation subspaces:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0     
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
       Compute scaled dot-product attention.    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0)
        
        attention_weights = F.softmax(scores, dim=-1
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights

def demonstrate_multi_head_attention():
Demonstrate multi-head attention."""
    
    # Parameters
    batch_size = 1   seq_len = 8
    d_model = 64 n_heads = 8
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Forward pass
    output, attention_weights = mha(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(fAttention weights shape: {attention_weights.shape})  # Show attention weights for all heads
    print(f"\nAttention weights for all heads (batch 0    for i in range(n_heads):
        print(f"Head {i}:")
        print(attention_weights[0, i].detach().numpy())
        print()
    
    return output, attention_weights

# Run multi-head demonstration
mha_output, mha_weights = demonstrate_multi_head_attention()
```

## Computational Complexity Analysis

Let's analyze the computational complexity of self-attention:

```python
import time

def complexity_analysis():
    """Analyze computational complexity of self-attention."""
    
    seq_lengths = 1050100, 50]
    d_model = 64 n_heads = 8  
    times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, d_model)
        mha = MultiHeadAttention(d_model, n_heads)
        
        # Warm up
        for _ in range(10):
            _ = mha(x)
        
        # Time the operation
        start_time = time.time()
        for _ in range(10):
            _ = mha(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) /100      times.append(avg_time)
        
        print(f"Sequence length: {seq_len}, Average time: {avg_time:0.6)    # Plot results
    plt.figure(figsize=(10,6)
    plt.plot(seq_lengths, times, 'bo-')
    plt.xlabel('Sequence Length')
    plt.ylabel('Average Time (seconds))   plt.title('Self-Attention Computational Complexity')
    plt.grid(True)
    plt.show()
    
    # Theoretical complexity: O(n²)
    theoretical_times = t * (seq_lengths[0]**2) / (seq_lengths[0]**2) for t in times]
    
    plt.figure(figsize=(10,6)
    plt.plot(seq_lengths, times, 'bo-', label='Actual')
    plt.plot(seq_lengths, theoretical_times,ro-label=Theoretical O(n²)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Average Time (seconds))   plt.title('Self-Attention: Actual vs Theoretical Complexity')
    plt.legend()
    plt.grid(true)
    plt.show()

# Run complexity analysis
complexity_analysis()
```

## Summary

Self-attention is a powerful mechanism that:

1 **Captures Long-Range Dependencies**: Can directly attend to any position in the sequence
2. **Is Highly Parallelizable**: All attention computations can be done in parallel
3. **Learns Flexible Relationships**: The attention weights are learned from data
4. **Scales with Sequence Length**: Computational complexity is $`O(n^2 \cdot d)`$

The key mathematical components are:
- **Query-Key-Value Projections**: Transform input into three different representations
- **Scaled Dot-Product**: Compute compatibility scores with scaling for stability
- **Softmax Normalization**: Convert scores to probability distributions
- **Weighted Sum**: Combine values based on attention weights

Multi-head attention extends this by allowing the model to attend to different representation subspaces simultaneously, making it more expressive and powerful. 