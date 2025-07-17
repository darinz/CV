# 08 Model Architectures for Vision-Language

Vision-language models use various architectures to combine and process information from images and text. The main approaches are Early Fusion, Late Fusion, Attention Mechanisms, and Multimodal Transformers.

## 1. Early Fusion
- **Definition:** Combine image and text features early in the network, then process jointly.
- **Example:** Concatenate image and text embeddings, then feed to a neural network.

### Example: Early Fusion (PyTorch)
```python
import torch
import torch.nn as nn

class EarlyFusionModel(nn.Module):
    def __init__(self, img_dim, text_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, img_emb, text_emb):
        x = torch.cat([img_emb, text_emb], dim=-1)
        return self.fc(x)

img_emb = torch.randn(5, 128)
text_emb = torch.randn(5, 64)
model = EarlyFusionModel(img_dim=128, text_dim=64, hidden_dim=256, out_dim=10)
output = model(img_emb, text_emb)
print('Output shape:', output.shape)  # (5, 10)
```

## 2. Late Fusion
- **Definition:** Process each modality independently, then combine at a later stage (e.g., for decision making).
- **Example:** Separate encoders for image and text, combine outputs for classification.

### Example: Late Fusion (PyTorch)
```python
import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    def __init__(self, img_dim, text_dim, out_dim):
        super().__init__()
        self.img_fc = nn.Linear(img_dim, out_dim)
        self.text_fc = nn.Linear(text_dim, out_dim)
        self.classifier = nn.Linear(out_dim * 2, 1)
    def forward(self, img_emb, text_emb):
        img_out = self.img_fc(img_emb)
        text_out = self.text_fc(text_emb)
        x = torch.cat([img_out, text_out], dim=-1)
        return self.classifier(x)

img_emb = torch.randn(4, 128)
text_emb = torch.randn(4, 64)
model = LateFusionModel(img_dim=128, text_dim=64, out_dim=32)
output = model(img_emb, text_emb)
print('Output shape:', output.shape)  # (4, 1)
```

## 3. Attention Mechanisms
- **Co-attention:** Attend to relevant regions in both image and text.
- **Cross-attention:** Use one modality to attend to another (e.g., text attends to image regions).

### Example: Cross-Attention (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
    def forward(self, x, context):
        # x: (batch, seq_len1, d_model), context: (batch, seq_len2, d_model)
        Q = self.query(x)
        K = self.key(context)
        V = self.value(context)
        attn = F.softmax(Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5), dim=-1)
        return attn @ V

x = torch.randn(2, 5, 64)      # e.g., text tokens
context = torch.randn(2, 10, 64)  # e.g., image regions
attn_layer = CrossAttention(d_model=64)
output = attn_layer(x, context)
print('Output shape:', output.shape)  # (2, 5, 64)
```

## 4. Multimodal Transformers
- **Definition:** Jointly process image patches and text tokens using transformer layers.
- **Examples:** ViLBERT, LXMERT, CLIP, BLIP.

### Example: Multimodal Transformer (Conceptual)
```python
# Pseudocode for multimodal transformer input
image_patches = ...  # (batch, num_patches, d_model)
text_tokens = ...    # (batch, seq_len, d_model)
inputs = torch.cat([image_patches, text_tokens], dim=1)
# Feed to transformer encoder
outputs = transformer_encoder(inputs)
```

## Summary
Model architectures for vision-language tasks range from simple fusion to advanced multimodal transformers, each with trade-offs in flexibility and performance. 