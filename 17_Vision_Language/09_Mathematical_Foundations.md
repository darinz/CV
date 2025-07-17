# 09 Mathematical Foundations for Vision-Language

Vision-language models rely on mathematical principles to align and relate information across modalities. Two key foundations are Multimodal Embedding Alignment and Contrastive Learning.

## 1. Multimodal Embedding Alignment
- **Goal:** Map images and text to a shared embedding space so that paired samples are close, unpaired are far apart.
- **Loss Function:**
```math
L_{align} = \| f_{img}(I) - f_{text}(T) \|^2
```
- **Explanation:** Minimize the distance between embeddings of paired image-text samples.

### Example: Embedding Alignment (PyTorch)
```python
import torch

img_emb = torch.randn(6, 128)
txt_emb = torch.randn(6, 128)
loss = ((img_emb - txt_emb) ** 2).mean()
print('Alignment loss:', loss.item())
```

## 2. Contrastive Learning for Vision-Language
- **Goal:** Bring paired samples together, push unpaired apart using a contrastive loss.
- **Loss Function (InfoNCE):**
```math
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, t_j)/\tau)}
```
where $v_i$ and $t_i$ are image and text embeddings for the $i$-th pair, $\tau$ is a temperature parameter.

### Example: Contrastive Loss (PyTorch)
```python
import torch
import torch.nn.functional as F

def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a @ b.T)

img_emb = torch.randn(8, 128)
txt_emb = torch.randn(8, 128)
sim_matrix = cosine_sim(img_emb, txt_emb)
target = torch.arange(8)
loss = F.cross_entropy(sim_matrix, target)
print('Contrastive loss:', loss.item())
```

## Summary
Mathematical foundations like embedding alignment and contrastive learning are essential for training effective vision-language models. 