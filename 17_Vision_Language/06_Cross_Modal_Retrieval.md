# 06 Cross-Modal Retrieval

Cross-modal retrieval is the task of retrieving relevant data in one modality (e.g., images) given a query in another modality (e.g., text), and vice versa. This is a key application of joint vision-language models.

## Problem Definition
- **Text-to-Image Retrieval:** Given a text query, retrieve relevant images.
- **Image-to-Text Retrieval:** Given an image, retrieve relevant text descriptions.

## Joint Embedding Space
- Learn $f_{img}(I)$ and $f_{text}(T)$ so that paired image-text samples are close in embedding space.

### Contrastive Loss
A common approach is to use a contrastive loss to bring paired samples together and push unpaired samples apart:

```math
L = -\log \frac{\exp(\text{sim}(f_{img}(I), f_{text}(T)) / \tau)}{\sum_{T'} \exp(\text{sim}(f_{img}(I), f_{text}(T')) / \tau)}
```
where $\text{sim}$ is a similarity function (e.g., dot product, cosine), and $\tau$ is a temperature parameter.

## Example: Contrastive Learning for Cross-Modal Retrieval (PyTorch)

```python
import torch
import torch.nn.functional as F

def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a @ b.T)

# Dummy embeddings for 8 image-text pairs
img_emb = torch.randn(8, 128)
txt_emb = torch.randn(8, 128)

# Compute similarity matrix
sim_matrix = cosine_sim(img_emb, txt_emb)  # (8, 8)

# Contrastive loss (InfoNCE)
target = torch.arange(8)
loss_i2t = F.cross_entropy(sim_matrix, target)
loss_t2i = F.cross_entropy(sim_matrix.T, target)
loss = (loss_i2t + loss_t2i) / 2
print('Contrastive loss:', loss.item())
```

### Explanation
- **cosine_sim:** Computes cosine similarity between all image-text pairs.
- **sim_matrix:** Similarity scores for all pairs in the batch.
- **loss:** Encourages correct pairs to have higher similarity than incorrect pairs.

## Real-World Models
- **CLIP, ALIGN:** Large-scale contrastive pretraining for cross-modal retrieval.

## Summary
Cross-modal retrieval enables searching across modalities, powered by joint embedding models and contrastive learning. 