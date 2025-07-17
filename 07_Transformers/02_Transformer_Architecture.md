# Transformer Architecture

The Transformer architecture, introduced in the seminal paper "Attention Is All You Need," revolutionized natural language processing by replacing recurrent neural networks with a purely attention-based approach. This architecture has become the foundation for state-of-the-art models like BERT, GPT, and T5.

## Architecture Overview

The Transformer consists of an encoder and decoder, each containing multiple identical layers. Unlike RNNs, Transformers process entire sequences in parallel, making them highly efficient for training and inference.

### Key Components

1. **Encoder**: Processes the input sequence
2. **Decoder**: Generates the output sequence
3. **Multi-Head Self-Attention**: Captures relationships between all positions
4. **Position-wise Feed-Forward Networks**: Processes each position independently
5. **Layer Normalization**: Stabilizes training
6. **Positional Encoding**: Injects positional information

## Mathematical Foundation

### Encoder Layer

Each encoder layer consists of two sub-layers:

```math
\text{EncoderLayer}(x) = \text{LayerNorm}(\text{FFN}(\text{LayerNorm}(x + \text{MultiHead}(x))) + \text{MultiHead}(x))
```

### Decoder Layer

Each decoder layer consists of three sub-layers:

```math
\text{DecoderLayer}(x, y) = \text{LayerNorm}(\text{FFN}(\text{LayerNorm}(y + \text{CrossAttention}(y, x))) + \text{CrossAttention}(y, x))
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

Each layer contains a position-wise feed-forward network:

```math
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
```

where $`W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}`$, $`W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}`$.

### Layer Normalization

Layer normalization normalizes across the feature dimension:

```math
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

where $`\mu`$ and $`\sigma^2`$ are computed across the feature dimension.

## Python Implementation

Let's implement the complete Transformer architecture:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0 / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0     
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection
        attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_len=500, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder
        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])
        
        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def generate_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0)
        return tgt_mask
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        for layer in self.encoder:
            src = layer(src, src_mask)
        
        return src
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        for layer in self.decoder:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        
        return tgt
    
    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        output = self.output_projection(dec_output)
        return output

# Example usage
def demonstrate_transformer():
    # Parameters
    src_vocab_size = 1000    tgt_vocab_size = 10
    d_model = 512   n_heads = 8
    n_layers =6 d_ff = 2048
    
    # Create model
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff)
    
    # Create sample data
    batch_size = 2
    src_len = 10 tgt_len = 8
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    output = transformer(src, tgt)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    return transformer, output

# Run demonstration
if __name__ == "__main__":
    model, output = demonstrate_transformer()

## Positional Encoding Analysis

Let's analyze how positional encoding works:

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_positional_encoding():
    """Analyze positional encoding patterns."""
    d_model = 64
    max_len = 100   
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                       (-math.log(1000.0 / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Visualize positional encoding
    plt.figure(figsize=(128, 10))
    plt.imshow(pe.numpy(), cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Heatmap')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.show()
    
    # Show specific dimensions
    plt.figure(figsize=(15, 10))
    for i in range(8):
        plt.subplot(2, i+1)
        plt.plot(pe[:, i].numpy())
        plt.title(f'Dimension {i}')
        plt.xlabel('Position')
        plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
    # Analyze relative positions
    def get_relative_position_encoding(pos1, pos2, d_model):
        """Calculate relative position encoding between two positions."""
        rel_pos = pos1 - pos2        encoding = torch.zeros(d_model)
        
        for i in range(0, d_model, 2):
            freq = 10.0 / (1000.0 ** (i / d_model))
            encoding[i] = math.sin(rel_pos * freq)
            if i + 1 < d_model:
                encoding[i+1] = math.cos(rel_pos * freq)
        
        return encoding
    
    # Show relative positions
    positions = [0, 1, 5, 10, 15, 20, 25, 30]
    plt.figure(figsize=(12, 8))
    
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            rel_encoding = get_relative_position_encoding(pos1, pos2, d_model)
            plt.subplot(len(positions), len(positions), i * len(positions) + j + 1)
            plt.plot(rel_encoding.numpy())
            plt.title(f'Pos {pos1} to Pos {pos2}')
            plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()

# Run positional encoding analysis
analyze_positional_encoding()
```

## Layer Normalization vs Batch Normalization

Let's compare different normalization techniques:

```python
def compare_normalization():
    """Compare different normalization techniques."""
    batch_size = 4
    seq_len = 10
    d_model = 64
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Layer Normalization (normalizes across features)
    layer_norm = nn.LayerNorm(d_model)
    layer_output = layer_norm(x)
    
    # Batch Normalization (normalizes across batch)
    batch_norm = nn.BatchNorm1d(d_model)
    batch_output = batch_norm(x.transpose(1, 2).transpose(1, 2))
    
    print("Layer Normalization:")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {layer_output.mean():.4f}, std: {layer_output.std():.4f}")
    
    print("\nBatch Normalization:")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {batch_output.mean():.4f}, std: {batch_output.std():.4f}")
    
    # Visualize distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(x.flatten().numpy(), bins=50, alpha=0.7, label='Input')
    plt.title('Input Distribution')
    plt.legend()
    
    plt.subplot(132)
    plt.hist(layer_output.flatten().numpy(), bins=50, alpha=0.7, label='Layer Norm')
    plt.title('Layer Normalization')
    plt.legend()
    
    plt.subplot(133)
    plt.hist(batch_output.flatten().numpy(), bins=50, alpha=0.7, label='Batch Norm')
    plt.title('Batch Normalization')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return layer_output, batch_output

# Run normalization comparison
layer_out, batch_out = compare_normalization()
```

## Training Process

Let's implement a complete training loop for the Transformer:

```python
def create_sample_data(vocab_size, num_samples=10, max_len=20):
    """Create synthetic training data."""
    src_data = []
    tgt_data = []
    
    for _ in range(num_samples):
        # Create source sequence (add 1 to avoid 0 which is padding)
        src_len = torch.randint(5, max_len, (1,)).item()
        src = torch.randint(1, vocab_size//2, (src_len,))
        
        # Create target sequence (slightly different vocabulary)
        tgt_len = torch.randint(5, max_len, (1,)).item()
        tgt = torch.randint(vocab_size//2, vocab_size, (tgt_len,))
        
        src_data.append(src)
        tgt_data.append(tgt)
    
    return src_data, tgt_data

def pad_sequences(sequences, max_len=None):
    """Pad sequences to the same length."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=seq.dtype)])
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)
    
    return torch.stack(padded)

def train_transformer():
    """Train a simple Transformer model."""
    
    # Parameters
    src_vocab_size = 1000    tgt_vocab_size = 10
    d_model = 128 # Smaller for faster training
    n_heads = 4
    n_layers = 2    d_ff = 512
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Create data
    src_data, tgt_data = create_sample_data(src_vocab_size, num_samples=500)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0        
        # Create batches
        for i in range(0, len(src_data), batch_size):
            batch_src = src_data[i:i+batch_size]
            batch_tgt = tgt_data[i:i+batch_size]
            
            # Pad sequences
            src = pad_sequences(batch_src)
            tgt = pad_sequences(batch_tgt)
            
            # Prepare target for loss calculation
            tgt_input = tgt[:, :-1] # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            # Reshape for loss calculation
            output = output.view(-1, tgt_vocab_size)
            tgt_output = tgt_output.view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1      
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")   # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model, losses

# Run training (commented out for brevity)
# model, losses = train_transformer()
```

## Attention Visualization

Let's create a function to visualize attention patterns:

```python
def visualize_attention_patterns(model, src, tgt):
    """Visualize attention patterns in the Transformer."""
    model.eval()
    with torch.no_grad():
        # Get attention weights from each layer
        src_mask = model.generate_src_mask(src)
        tgt_mask = model.generate_tgt_mask(tgt)
        
        # Encode
        enc_output = model.encode(src, src_mask)
        
        # Decode and collect attention weights
        tgt_input = tgt[:, :-1]
        tgt_embed = model.tgt_embedding(tgt_input) * math.sqrt(model.d_model)
        tgt_embed = model.pos_encoding(tgt_embed)
        tgt_embed = model.dropout(tgt_embed)
        
        attention_weights = []
        
        for layer in model.decoder:
            # Self-attention
            attn_output, self_attn_weights = layer.self_attention(
                tgt_embed, tgt_embed, tgt_embed, tgt_mask
            )
            tgt_embed = layer.norm1(tgt_embed + model.dropout(attn_output))
            
            # Cross-attention
            attn_output, cross_attn_weights = layer.cross_attention(
                tgt_embed, enc_output, enc_output, src_mask
            )
            tgt_embed = layer.norm2(tgt_embed + model.dropout(attn_output))
            
            # Feed-forward
            ff_output = layer.feed_forward(tgt_embed)
            tgt_embed = layer.norm3(tgt_embed + model.dropout(ff_output))
            
            attention_weights.append({
               'self_attention': self_attn_weights,
                'cross_attention': cross_attn_weights
            })
    
    return attention_weights

def plot_attention_heatmaps(attention_weights, src_tokens=None, tgt_tokens=None):
    """Plot attention heatmaps for all layers."""
    
    n_layers = len(attention_weights)
    fig, axes = plt.subplots(n_layers, 2, figsize=(15, 5 * n_layers))
    
    for layer_idx, layer_weights in enumerate(attention_weights):
        # Self-attention heatmap
        self_attn = layer_weights['self_attention'][0, 0].numpy()  # First batch, first head
        axes[layer_idx, 0].imshow(self_attn, cmap='Blues')
        axes[layer_idx, 0].set_title(f'Layer {layer_idx+1} - Self Attention')
        axes[layer_idx, 0].set_xlabel('Key Position')
        axes[layer_idx, 0].set_ylabel('Query Position')
        
        # Cross-attention heatmap
        cross_attn = layer_weights['cross_attention'][0, 0].numpy()  # First batch, first head
        axes[layer_idx, 1].imshow(cross_attn, cmap='Reds')
        axes[layer_idx, 1].set_title(f'Layer {layer_idx+1} - Cross Attention')
        axes[layer_idx, 1].set_xlabel('Source Position')
        axes[layer_idx, 1].set_ylabel('Target Position')
    
    plt.tight_layout()
    plt.show()

# Example usage
def demonstrate_attention_visualization():
    # Create a simple model
    model = Transformer(100, 10, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    
    # Create sample input
    src = torch.randint(1, 100, (1, 8))
    tgt = torch.randint(1, 100, (1, 6))
    
    # Get attention weights
    attention_weights = visualize_attention_patterns(model, src, tgt)
    
    # Plot attention heatmaps
    plot_attention_heatmaps(attention_weights)
    
    return attention_weights

# Run attention visualization
# attention_weights = demonstrate_attention_visualization()
```

## Summary

The Transformer architecture represents a paradigm shift in sequence modeling:

1. **Parallel Processing**: Unlike RNNs, Transformers process entire sequences in parallel
2. **Self-Attention**: Captures relationships between all positions in the sequence
3. **Positional Encoding**: Injects positional information without recurrence
4. **Layer Normalization**: Stabilizes training by normalizing across features
5. **Residual Connections**: Helps with gradient flow in deep networks

Key mathematical components:
- **Multi-Head Attention**: Allows attending to different representation subspaces
- **Position-wise Feed-Forward**: Processes each position independently
- **Positional Encoding**: Uses sinusoidal functions to encode position information
- **Layer Normalization**: Normalizes across the feature dimension

The architecture has become the foundation for modern language models and has been successfully adapted for various tasks including machine translation, text generation, and even computer vision tasks. 