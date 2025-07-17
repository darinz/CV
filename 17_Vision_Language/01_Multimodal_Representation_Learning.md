# 01 Multimodal Representation Learning

Multimodal representation learning is the process of learning joint feature representations from multiple modalities, such as images and text. This is foundational for tasks where understanding both visual and linguistic information is crucial (e.g., image captioning, visual question answering).

## Why Multimodal Representation?

- **Single-modality models** (e.g., only images or only text) cannot capture relationships between modalities.
- **Multimodal models** learn to align and fuse information, enabling richer understanding and cross-modal tasks.

## Core Idea

Given paired data $(I, T)$ (image $I$ and text $T$), learn functions $f_{img}(I)$ and $f_{text}(T)$ that map both to a shared embedding space.

### Mathematical Formulation

- $f_{img}(I)$: Image encoder (e.g., CNN, Vision Transformer)
- $f_{text}(T)$: Text encoder (e.g., RNN, Transformer, BERT)
- **Goal:** Paired $(I, T)$ should be close in embedding space; unpaired should be far apart.

**Alignment Loss Example:**
```math
L_{align} = \| f_{img}(I) - f_{text}(T) \|^2
```

## Example: Simple Joint Embedding with PyTorch

Below is a minimal example using PyTorch to learn joint embeddings for images and text.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy image encoder (e.g., a small CNN)
class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, output_dim)  # For 32x32 RGB images
        )
    def forward(self, x):
        return self.encoder(x)

# Dummy text encoder (e.g., a simple embedding + mean pooling)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, output_dim)
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        pooled = emb.mean(dim=1)  # Mean pooling
        return self.fc(pooled)

# Example training loop
image_encoder = ImageEncoder(output_dim=128)
text_encoder = TextEncoder(vocab_size=1000, embed_dim=64, output_dim=128)

optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-3)

# Dummy data: batch of 16 images and paired text
images = torch.randn(16, 3, 32, 32)  # 16 RGB images
texts = torch.randint(0, 1000, (16, 10))  # 16 text samples, 10 tokens each

for epoch in range(100):
    img_emb = image_encoder(images)
    txt_emb = text_encoder(texts)
    # Alignment loss: MSE between paired embeddings
    loss = ((img_emb - txt_emb) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Explanation
- **ImageEncoder**: Flattens and projects image to embedding.
- **TextEncoder**: Embeds tokens, mean-pools, projects to embedding.
- **Loss**: Mean squared error between paired image and text embeddings.

## Real-World Models
- **CLIP** (Contrastive Language-Image Pretraining): Uses contrastive loss to align image and text embeddings.
- **ALIGN**: Similar approach, large-scale data.

## Summary
Multimodal representation learning enables models to understand and relate information across modalities, forming the basis for many vision-language tasks. 