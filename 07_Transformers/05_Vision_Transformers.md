# Vision Transformers (ViT)

Vision Transformers (ViT) adapt the Transformer architecture for computer vision tasks by treating images as sequences of patches. This approach has revolutionized computer vision, achieving state-of-the-art performance on various tasks.

## Core Concept

Instead of processing images pixel by pixel, ViT divides images into fixed-size patches and treats each patch as a token in a sequence. This allows the application of the powerful self-attention mechanism to visual data.

### Patch Embedding

The image is divided into $`N`$ patches of size $`P \times P`$:

```math
N = \frac{H \times W}{P^2}
```

where $`H`$ and $`W`$ are the height and width of the image.

Each patch is flattened and projected to a $`d_{model}`$-dimensional embedding:

```math
x_i = \text{Linear}(\text{Flatten}(P_i))
```

where $`P_i`$ is the $`i`$-th patch.

### Position Embedding

Learnable position embeddings are added to the patch embeddings:

```math
z_0 = [x_{class}; x_1, x_2, \ldots, x_n] + E_{pos}
```

where $`x_{class}`$ is a learnable class token and $`E_{pos}`$ are position embeddings.

## Mathematical Foundation

### Patch Embedding Process

For an input image $`I \in \mathbb{R}^{H \times W \times C}`$:

1. **Patch Division**: Divide into $`N`$ patches of size $`P \times P \times C`$
2. **Flattening**: Reshape each patch to $`P^2C`$-dimensional vector
3. **Linear Projection**: Project to $`d_{model}`$-dimensional space

```math
x_i = \text{Linear}(\text{Flatten}(I_i)) + E_{pos}^i
```

### Self-Attention in Vision

The self-attention mechanism remains the same as in language models:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

However, now $`Q`$, $`K`$, and $`V`$ represent patches rather than words.

### Classification Head

For classification tasks, the class token from the final layer is used:

```math
y = \text{MLP}(\text{LayerNorm}(z_L^0))
```

where $`z_L^0`$ is the class token at the final layer $`L`$.

## Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size = x.size(0)
        
        # Apply convolution to get patch embeddings
        # Output: (batch_size, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        
        # Flatten spatial dimensions
        # Output: (batch_size, embed_dim, n_patches)
        x = x.flatten(2)
        
        # Transpose to get (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches+1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Extract class token for classification
        cls_token = x[:, 0]
        
        # Classification head
        output = self.head(cls_token)
        
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.mlp(x)
        x = self.norm2(x + ff_out)
        
        return x

def demonstrate_vision_transformer():
    """Demonstrate Vision Transformer."""
    
    # Parameters
    batch_size = 2    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 1000   embed_dim = 768    depth = 12    num_heads = 12
    
    # Create model
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )
    
    # Create sample input
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # Forward pass
    output = vit(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of patches: {vit.n_patches}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Embedding dimension: {embed_dim}")   
    return vit, output

# Run demonstration
vit_model, vit_output = demonstrate_vision_transformer()
```

## Patch Visualization

Let's visualize how images are divided into patches:

```python
def visualize_patches(image, patch_size=16):
    """Visualize how an image is divided into patches."""
    
    # Create a sample image
    if image is None:
        # Create a simple test image
        image = np.random.rand(224, 224, 3)
    
    height, width = image.shape[:2]
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size
    
    # Create figure
    fig, axes = plt.subplots(n_patches_h, n_patches_w, figsize=(12,12))
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Extract patch
            patch = image[i*patch_size:(i+1)*patch_size, 
                         j*patch_size:(j+1)*patch_size]
            
            # Display patch
            axes[i, j].imshow(patch)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Patch ({i},{j})')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Image size: {height}x{width}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Number of patches: {n_patches_h * n_patches_w}")

# Create a sample image and visualize patches
sample_image = np.random.rand(224, 224, 3)
visualize_patches(sample_image, patch_size=16)
```

## Attention Visualization

Let's visualize attention patterns in Vision Transformers:

```python
class VisionTransformerWithAttention(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(VisionTransformerWithAttention, self).__init__(*args, **kwargs)
        self.attention_weights = []
    
    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Pass through transformer encoder and collect attention weights
        attention_weights = []
        for layer in self.encoder_layers:
            # Get attention weights from the layer
            attn_out, attn_weights = layer.attention(x, x, x)
            attention_weights.append(attn_weights)
            
            # Apply attention
            x = layer.norm1(x + attn_out)
            
            # Feed-forward
            ff_out = layer.mlp(x)
            x = layer.norm2(x + ff_out)
        
        # Layer normalization
        x = self.norm(x)
        
        # Extract class token
        cls_token = x[:, 0]
        
        # Classification head
        output = self.head(cls_token)
        
        if return_attention:
            return output, attention_weights
        return output

def visualize_attention_patterns(model, image, layer_idx=0, head_idx=0):
    """Visualize attention patterns for a specific layer and head."""
    model.eval()
    with torch.no_grad():
        # Forward pass with attention weights
        output, attention_weights = model(image, return_attention=True)
        
        # Get attention weights for specified layer and head
        attn_weights = attention_weights[layer_idx][0, head_idx].numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_weights, cmap='Blues')
        plt.colorbar()
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.show()
        
        # Show attention from class token to patches
        class_to_patches = attn_weights[0, 1] # Class token attends to patches
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(class_to_patches)), class_to_patches)
        plt.title(f'Class Token Attention to Patches - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Patch Index')
        plt.ylabel('Attention Weight')
        plt.show()
        
        return attn_weights

# Create model with attention visualization
vit_with_attention = VisionTransformerWithAttention(
    img_size=224, patch_size=16, embed_dim=768, depth=6, num_heads=12
)

# Create sample input
sample_input = torch.randn(1, 3, 224, 224)
visualize_attention_patterns(
    vit_with_attention, sample_input, layer_idx=0, head_idx=0
)
```

## Position Embedding Analysis

Let's analyze how position embeddings work in Vision Transformers:

```python
def analyze_position_embeddings(model):
    """Analyze position embeddings in Vision Transformer."""
    # Get position embeddings
    pos_embed = model.pos_embed.data
    
    # Visualize position embeddings
    plt.figure(figsize=(15, 10))
    
    # Show first few dimensions
    for i in range(16):
        plt.subplot(4, i+1)
        plt.imshow(pos_embed[0, :, i].reshape(14, 15), cmap='RdBu')
        plt.title(f'Dimension {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze similarity between positions
    similarity_matrix = torch.matmul(pos_embed[0], pos_embed[0].T)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix.numpy(), cmap='Blues')
    plt.colorbar()
    plt.title('Position Embedding Similarity Matrix')
    plt.xlabel('Position')
    plt.ylabel('Position') 
    plt.show()
    
    # Show similarity between adjacent positions
    adjacent_similarities = []
    for i in range(similarity_matrix.size(0) - 1):
        adjacent_similarities.append(similarity_matrix[i, i+1].item())
    
    plt.figure(figsize=(10, 6))
    plt.plot(adjacent_similarities)
    plt.title('Similarity Between Adjacent Positions')
    plt.xlabel('Position Index')
    plt.ylabel('Similarity')
    plt.grid(True)
    plt.show()
    
    return pos_embed, similarity_matrix

# Analyze position embeddings
pos_embed, similarity_matrix = analyze_position_embeddings(vit_model)
```

## Training Vision Transformers

Let's implement training for Vision Transformers:

```python
def train_vision_transformer():
    """Demonstrate training Vision Transformer."""
    # Model parameters
    img_size = 224
    patch_size = 16   embed_dim = 256 # Smaller for faster training
    depth = 6
    num_heads = 8
    num_classes = 10
    
    # Create model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_classes=num_classes
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(10):
        epoch_loss = 0
        num_batches = 0      
        for batch in range(10):  # Simulate batches
            # Create dummy data
            images = torch.randn(8, 3, img_size, img_size)
            labels = torch.randint(0, num_classes, (8,))
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1        
        # Update learning rate
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/10 Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")   # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Vision Transformer Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return model, losses

# Run training demonstration
# trained_model, training_losses = train_vision_transformer()
```

## Comparison with CNNs

Let's compare Vision Transformers with traditional CNNs:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1, 1)
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def compare_vision_models():
    """Compare Vision Transformer with CNN."""
    # Model parameters
    img_size = 224
    num_classes = 10
    batch_size = 4
    
    # Create models
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
        num_classes=num_classes
    )
    
    cnn = SimpleCNN(num_classes=num_classes)
    
    # Create sample input
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Compare outputs
    vit_output = vit(x)
    cnn_output = cnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"ViT output shape: {vit_output.shape}")
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Compare parameter counts
    vit_params = sum(p.numel() for p in vit.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())
    
    print(f"ViT parameters: {vit_params:,}")
    print(f"CNN parameters: {cnn_params:,}")
    print(f"Parameter ratio (ViT/CNN): {vit_params/cnn_params:.2f}")
    
    # Compare computational complexity
    import time
    
    # Time ViT
    vit.eval()
    start_time = time.time()
    for _ in range(10):
        _ = vit(x)
    vit_time = (time.time() - start_time) / 10   
    # Time CNN
    cnn.eval()
    start_time = time.time()
    for _ in range(10):
        _ = cnn(x)
    cnn_time = (time.time() - start_time) / 10    
    print(f"\nViT inference time: {vit_time:0.4e} seconds")
    print(f"CNN inference time: {cnn_time:0.4e} seconds")
    print(f"Speed ratio (CNN/ViT): {cnn_time/vit_time:.2f}")   
    return vit, cnn, vit_output, cnn_output

# Run comparison
vit_model, cnn_model, vit_out, cnn_out = compare_vision_models()
```

## Summary

Vision Transformers represent a paradigm shift in computer vision:
1. **Patch-based Processing**: Images are divided into patches and treated as sequences
2. **Self-Attention**: Captures global relationships between all patches
3. **Position Embeddings**: Learnable position information for spatial relationships
4. **Scalability**: Can handle variable input sizes and scales well with data

Key advantages:
- **Global Attention**: Unlike CNNs, can attend to any patch from any position
- **Parallel Processing**: All patches are processed simultaneously
- **Transfer Learning**: Pre-trained models can be fine-tuned for various tasks
- **Interpretability**: Attention weights provide insights into model decisions

Key mathematical components:
- **Patch Embedding**: Linear projection of flattened image patches
- **Position Embedding**: Learnable embeddings for spatial positions
- **Self-Attention**: Same mechanism as language models, applied to patches
- **Classification Head**: MLP applied to the class token

Vision Transformers have achieved state-of-the-art performance on various computer vision tasks and continue to be an active area of research. 