# Attention Variants

While the original scaled dot-product attention is powerful, researchers have developed various attention variants to address specific challenges and improve performance. This guide explores the most important attention variants and their applications.

## Masked Self-Attention

Masked self-attention is crucial for decoder layers in sequence-to-sequence tasks, preventing the model from attending to future positions during training.

### Mathematical Formulation

For decoder layers, attention is masked to prevent attending to future positions:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

where $M_{ij} = -\infty$ if $i > j$, and $0$ otherwise.

The mask matrix $M$ is defined as:

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & \cdots & -\infty \\
0 & 0 & -\infty & \cdots & -\infty \\
0 & 0 & 0 & \cdots & -\infty \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 0
\end{bmatrix}
$$

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=00.1):
        super(MaskedSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_causal_mask(self, seq_len):
      Create causal mask for decoder attention.      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1t('-inf))       return mask
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        causal_mask = self.create_causal_mask(seq_len).to(scores.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0-inf)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(context)
        
        return output, attention_weights

def demonstrate_masked_attention():
Demonstrate masked self-attention."""
    
    # Parameters
    batch_size = 1   seq_len = 6
    d_model = 64 n_heads = 4
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize masked attention
    masked_attention = MaskedSelfAttention(d_model, n_heads)
    
    # Forward pass
    output, attention_weights = masked_attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(fAttention weights shape: {attention_weights.shape})  # Show attention weights for first head
    print(f"\nAttention weights (batch0ad 0):")
    print(attention_weights[0,0detach().numpy())
    
    # Verify causal property
    causal_mask = masked_attention.create_causal_mask(seq_len)
    print(fnCausal mask:")
    print(causal_mask.numpy())
    
    return output, attention_weights

# Run demonstration
output, attention_weights = demonstrate_masked_attention()
```

## Relative Positional Encoding

Relative positional encoding allows the model to understand relative positions between tokens rather than absolute positions.

### Mathematical Formulation

Instead of absolute positions, use relative positions:

$$
\text{RelativeAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + R\right)V
$$

where $R_{ij}$ encodes the relative position between $i$ and $j$:

$$
R_{ij} = \sum_{k=1}^{d_k} w_k \cdot \text{PE}(i-j)_k
$$

The relative positional encoding is computed as:

$$
\text{PE}(rel\_pos)_{2i} = \sin\left(\frac{rel\_pos}{1000^{2i/d_k}}\right)
$$

$$
\text{PE}(rel\_pos)_{2i+1} = \cos\left(\frac{rel\_pos}{1000^{2i/d_k}}\right)
$$

### Python Implementation

```python
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position=32    super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.relative_attention_bias = nn.Embedding(vocab_size, d_model)
        
        # Initialize weights
        nn.init.normal_(self.relative_attention_bias.weight,00.02  
    def forward(self, seq_len):
 Generate relative position embeddings."""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get embeddings
        embeddings = self.relative_attention_bias(final_mat)
        return embeddings

class RelativeSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_relative_position=32pout=0.1    super(RelativeSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_attention_bias = RelativePositionalEncoding(
            d_model, max_relative_position
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute content-based attention scores
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Get relative position embeddings
        relative_pos_embeddings = self.relative_attention_bias(seq_len)
        relative_pos_embeddings = relative_pos_embeddings.view(
            seq_len, seq_len, self.n_heads, self.d_k
        ).transpose(0,1).transpose(1, 2)
        
        # Compute position-based attention scores
        position_scores = torch.matmul(Q, relative_pos_embeddings.transpose(-2)
        
        # Combine content and position scores
        scores = content_scores + position_scores
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(context)
        
        return output, attention_weights

def demonstrate_relative_attention():
monstrate relative positional attention."""
    
    # Parameters
    batch_size = 1   seq_len = 8
    d_model = 64 n_heads = 4
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize relative attention
    relative_attention = RelativeSelfAttention(d_model, n_heads)
    
    # Forward pass
    output, attention_weights = relative_attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(fAttention weights shape: {attention_weights.shape})  # Show attention weights for first head
    print(f"\nAttention weights (batch0ad 0):")
    print(attention_weights[0,0detach().numpy())
    
    return output, attention_weights

# Run demonstration
relative_output, relative_weights = demonstrate_relative_attention()
```

## Sparse Attention

Sparse attention reduces computational complexity by attending to a subset of positions.

### Mathematical Formulation

Reduce computational complexity by attending to a subset of positions:

$$
\text{SparseAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M\right)V
$$

where $M$ is a sparse mask that determines which positions to attend to.

### Types of Sparse Attention

1. **Local Attention**: Attend only to nearby positions
2. **Strided Attention**: Attend to every k-th position
3. **Fixed Attention**: Attend to predefined positions

### Python Implementation

```python
class SparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_type='local', 
                 local_window=8, stride=2, dropout=00.1):
        super(SparseAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.attention_type = attention_type
        self.local_window = local_window
        self.stride = stride
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_local_mask(self, seq_len):
     te local attention mask.      mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2
            end = min(seq_len, i + self.local_window // 2)
            maski, start:end] = 1
        return mask
    
    def create_strided_mask(self, seq_len):
        strided attention mask.      mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Attend to every stride-th position
            for j in range(0, seq_len, self.stride):
                mask[i, j] = 1
            # Always attend to current position
            mask[i, i] = 1
        return mask
    
    def create_fixed_mask(self, seq_len):
     te fixed attention mask (example: attend to first, middle, last).      mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            # Attend to first position
            mask[i, 0 = 1
            # Attend to middle position
            mask[i, seq_len // 2 = 1
            # Attend to last position
            mask[i, -1 = 1
            # Attend to current position
            mask[i, i] = 1
        return mask
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Create sparse mask
        if self.attention_type == 'local':
            sparse_mask = self.create_local_mask(seq_len)
        elif self.attention_type == 'strided':
            sparse_mask = self.create_strided_mask(seq_len)
        elif self.attention_type == 'fixed':
            sparse_mask = self.create_fixed_mask(seq_len)
        else:
            sparse_mask = torch.ones(seq_len, seq_len)
        
        # Apply sparse mask
        sparse_mask = sparse_mask.to(scores.device)
        scores = scores.masked_fill(sparse_mask == 0-inf)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(context)
        
        return output, attention_weights

def demonstrate_sparse_attention():
onstrate different types of sparse attention."""
    
    # Parameters
    batch_size = 1
    seq_len =12
    d_model = 64 n_heads = 4
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test different sparse attention types
    attention_types = ['local',strided',fixed']
    
    for attention_type in attention_types:
        print(f"\n=== {attention_type.upper()} ATTENTION ===")
        
        # Initialize sparse attention
        sparse_attention = SparseAttention(
            d_model, n_heads, attention_type=attention_type
        )
        
        # Forward pass
        output, attention_weights = sparse_attention(x)
        
        print(f"Input shape: {x.shape})
        print(f"Output shape: {output.shape}")
        
        # Show attention weights for first head
        print(fAttention weights (batch 0, head 0):")
        print(attention_weights[0,0detach().numpy())
        
        # Show sparsity
        sparsity = (attention_weights00== 0loat().mean().item()
        print(f"Sparsity: [object Object]sparsity:.3f}")

# Run demonstration
demonstrate_sparse_attention()
```

## Linear Attention

Linear attention reduces the quadratic complexity of attention to linear complexity.

### Mathematical Formulation

Linear attention uses a feature map $\phi$ to approximate the softmax:

$$
\text{LinearAttention}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)\phi(K)^T}
$$

where $\phi$ is a feature map that approximates the exponential function.

### Python Implementation

```python
class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, feature_map=elu', dropout=0.1      super(LinearAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.feature_map = feature_map
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def apply_feature_map(self, x):
      Apply feature map to approximate softmax."        if self.feature_map == elu            return F.elu(x) + 1
        elif self.feature_map == 'relu':
            return F.relu(x)
        elif self.feature_map == 'softplus':
            return F.softplus(x)
        else:
            return x
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply feature map
        Q = self.apply_feature_map(Q)
        K = self.apply_feature_map(K)
        
        # Compute linear attention
        # Numerator: phi(Q) * (phi(K)^T * V)
        KV = torch.matmul(K.transpose(-2, -1)  # (batch, heads, d_k, d_k)
        numerator = torch.matmul(Q, KV)  # (batch, heads, seq_len, d_k)
        
        # Denominator: phi(Q) * phi(K)^T
        K_sum = K.sum(dim=-2, keepdim=True)  # (batch, heads, 1, d_k)
        denominator = torch.matmul(Q, K_sum.transpose(-2)  # (batch, heads, seq_len, 1)
        
        # Compute output
        context = numerator / (denominator + 1e-8)
        
        # Reshape and project back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(context)
        
        return output

def demonstrate_linear_attention():
Demonstrate linear attention."""
    
    # Parameters
    batch_size = 1
    seq_len =16
    d_model = 64 n_heads = 4
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize linear attention
    linear_attention = LinearAttention(d_model, n_heads, feature_map='elu')
    
    # Forward pass
    output = linear_attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compare with standard attention
    standard_attention = MaskedSelfAttention(d_model, n_heads)
    standard_output, _ = standard_attention(x)
    
    print(f"\nStandard attention output shape: {standard_output.shape}")
    
    # Compare outputs
    diff = torch.abs(output - standard_output).mean().item()
    print(f"Average difference: {diff:.6f})
    return output

# Run demonstration
linear_output = demonstrate_linear_attention()
```

## Multi-Query Attention

Multi-query attention reduces memory usage by sharing key and value projections across heads.

### Mathematical Formulation

In multi-query attention, only the query is projected separately for each head:

$$
Q_i = XW_i^Q, \quad K = XW_K, \quad V = XW_V
$$

$$
\text{head}_i = \text{Attention}(Q_i, K, V)
$$

### Python Implementation

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=00.1):
        super(MultiQueryAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Separate query projections for each head
        self.W_q = nn.ModuleList(         nn.Linear(d_model, self.d_k) for _ in range(n_heads)
        ])
        
        # Shared key and value projections
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Project queries for each head
        Q_heads = []
        for i in range(self.n_heads):
            Q_i = self.W_q[i](x)  # (batch, seq_len, d_k)
            Q_heads.append(Q_i)
        
        # Shared key and value projections
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention for each head
        context_heads = []
        for i in range(self.n_heads):
            Q_i = Q_heads[i].unsqueeze(1)  # (batch,1d_k)
            
            # Compute attention scores
            scores = torch.matmul(Q_i, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention weights to values
            context_i = torch.matmul(attention_weights, V)  # (batch,1, d_k)
            context_heads.append(context_i)
        
        # Concatenate heads
        context = torch.cat(context_heads, dim=1)  # (batch, n_heads, seq_len, d_k)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final projection
        output = self.W_o(context)
        
        return output

def demonstrate_multi_query_attention():
Demonstrate multi-query attention."""
    
    # Parameters
    batch_size = 1
    seq_len =10
    d_model = 64 n_heads = 4
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize multi-query attention
    mqa = MultiQueryAttention(d_model, n_heads)
    
    # Forward pass
    output = mqa(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compare parameter count with standard attention
    standard_attention = MaskedSelfAttention(d_model, n_heads)
    
    mqa_params = sum(p.numel() for p in mqa.parameters())
    standard_params = sum(p.numel() for p in standard_attention.parameters())
    
    print(f"\nMulti-Query Attention parameters: {mqa_params}")
    print(f"Standard Attention parameters: {standard_params}")
    print(f"Parameter reduction: {(1 - mqa_params/standard_params)*100)
    return output

# Run demonstration
mqa_output = demonstrate_multi_query_attention()
```

## Performance Comparison

Let's compare the computational complexity and memory usage of different attention variants:

```python
import time

def benchmark_attention_variants():
    """Benchmark different attention variants."""
    
    # Parameters
    batch_size =1
    seq_lengths =3264, 256
    d_model = 128 n_heads = 8
    
    variants = {
        Standard': MaskedSelfAttention(d_model, n_heads),
      Linear: LinearAttention(d_model, n_heads),
    Multi-Query': MultiQueryAttention(d_model, n_heads),
       Sparse (Local): SparseAttention(d_model, n_heads,local'),
       Sparse (Strided): SparseAttention(d_model, n_heads,strided')
    }
    
    results = {}
    
    for seq_len in seq_lengths:
        print(f\nSequence length: {seq_len}")
        x = torch.randn(batch_size, seq_len, d_model)
        
        for name, model in variants.items():
            # Warm up
            for _ in range(5):
                _ = model(x)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            if name not in results:
                results[name] = []
            results[name].append(avg_time)
            
            print(f" [object Object]name}: {avg_time:0.6)    # Plot results
    plt.figure(figsize=(12, 8))
    
    for name, times in results.items():
        plt.plot(seq_lengths, times, o- label=name)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Average Time (seconds))
    plt.title('Attention Variants Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log)  plt.show()
    
    return results

# Run benchmark
# results = benchmark_attention_variants()
```

## Summary

Attention variants address different challenges in Transformer architectures:

1 **Masked Self-Attention**: Prevents attending to future positions in decoders
2. **Relative Positional Encoding**: Captures relative positions more effectively
3. **Sparse Attention**: Reduces computational complexity by attending to subsets
4. **Linear Attention**: Achieves linear complexity with feature map approximation
5. **Multi-Query Attention**: Reduces memory usage by sharing key/value projections

Key trade-offs:
- **Standard Attention**: Full expressiveness but quadratic complexity
- **Linear Attention**: Linear complexity but approximation of softmax
- **Sparse Attention**: Reduced complexity but limited attention patterns
- **Multi-Query Attention**: Memory efficient but reduced expressiveness

The choice of attention variant depends on the specific requirements:
- **Long sequences**: Linear or sparse attention
- **Memory constraints**: Multi-query attention
- **Relative positions**: Relative positional encoding
- **Generation tasks**: Masked self-attention 