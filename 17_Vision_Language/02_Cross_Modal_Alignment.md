# 02 Cross-modal Alignment

Cross-modal alignment is the process of associating entities from different modalities, such as matching objects in an image to words in a sentence. This is crucial for tasks like image captioning, visual question answering, and cross-modal retrieval.

## What is Cross-modal Alignment?
- **Goal:** Ensure that semantically related visual and textual elements are close in a shared representation space.
- **Example:** The word "dog" in a caption should align with the region of the image containing a dog.

## Mathematical Formulation
Given image $I$ and text $T$:
- $f_{img}(I)$: Image encoder
- $f_{text}(T)$: Text encoder
- **Alignment Loss:**
```math
L_{align} = \| f_{img}(I) - f_{text}(T) \|^2
```

For fine-grained alignment (e.g., words to regions):
- $f_{img}(I) = [v_1, ..., v_K]$ (K image regions)
- $f_{text}(T) = [t_1, ..., t_L]$ (L words)
- **Attention-based alignment:**
```math
\alpha_{ij} = \text{softmax}(v_i^T W t_j)
```

## Example: Word-Region Alignment with Attention (PyTorch)

```python
import torch
import torch.nn.functional as F

# Suppose v: (batch, K, d), t: (batch, L, d)
v = torch.randn(8, 5, 128)  # 8 samples, 5 image regions, 128-dim
 t = torch.randn(8, 7, 128)  # 8 samples, 7 words, 128-dim

# Compute similarity matrix between regions and words
sim = torch.bmm(v, t.transpose(1, 2))  # (batch, K, L)

# Attention weights: for each region, attend to words
attn_weights = F.softmax(sim, dim=-1)  # (batch, K, L)

# Contextualized region features (weighted sum of word embeddings)
context = torch.bmm(attn_weights, t)  # (batch, K, 128)

# Alignment loss (e.g., MSE between region and context)
alignment_loss = ((v - context) ** 2).mean()
print('Alignment loss:', alignment_loss.item())
```

### Explanation
- **sim:** Computes similarity between each region and word.
- **attn_weights:** Softmax over words for each region.
- **context:** Weighted sum of word embeddings for each region.
- **alignment_loss:** Encourages regions to match their aligned word context.

## Real-World Use
- **Visual Grounding:** Aligning phrases to image regions.
- **VQA:** Aligning question words to relevant image parts.
- **CLIP:** Global alignment of image and text pairs.

## Summary
Cross-modal alignment enables models to associate and reason about related entities across modalities, which is essential for many vision-language tasks. 