# Image-to-Text Models

Image-to-text models generate textual descriptions from images. This guide covers BLIP, OFA, and Flamingo, with detailed math and code examples.

## 1. Introduction

Image-to-text generation is crucial for tasks like image captioning, visual question answering, and scene understanding. Modern models use transformers, cross-modal encoders, and multi-task learning.

## 2. BLIP (Bootstrapping Language-Image Pre-training)

BLIP uses separate encoders for images and text, followed by a cross-modal encoder.

### 2.1 Architecture

- **Image Encoder:** $`h_I = \text{ViT}(I)`$
- **Text Encoder:** $`h_T = \text{BERT}(T)`$
- **Cross-Modal Encoder:** $`h_{cross} = \text{CrossAttention}(h_T, h_I)`$

### 2.2 Training Objectives

- **Image-Text Contrastive:**
```math
L_{ITC} = -\log \frac{\exp(sim(I, T)/\tau)}{\sum_{T'} \exp(sim(I, T')/\tau)}
```
- **Image-Text Matching:**
```math
L_{ITM} = -\log P(y|I, T)
```
- **Language Modeling:**
```math
L_{LM} = -\sum_{t} \log P(w_t|w_{<t}, I)
```

#### Python Example: Contrastive Loss

```python
import torch
import torch.nn.functional as F

def blip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(len(image_embeds)).to(image_embeds.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
```

## 3. OFA (One-For-All)

OFA uses a unified transformer for both image and text tokens, enabling multi-task learning.

### 3.1 Unified Architecture

```math
h = \text{Transformer}([I_{tokens}, T_{tokens}])
```

### 3.2 Multi-Task Learning

```math
L = \lambda_1 L_{caption} + \lambda_2 L_{vqa} + \lambda_3 L_{grounding}
```

#### Python Example: Multi-Task Loss

```python
def multi_task_loss(loss_caption, loss_vqa, loss_grounding, lambdas):
    return lambdas[0]*loss_caption + lambdas[1]*loss_vqa + lambdas[2]*loss_grounding
```

## 4. Flamingo

Flamingo uses a perceiver resampler for images and gated cross-attention for fusion.

### 4.1 Perceiver Resampler

```math
h_I = \text{Perceiver}(I_{tokens})
```

### 4.2 Gated Cross-Attention

```math
h_{cross} = \text{GatedCrossAttention}(h_T, h_I)
```

#### Python Example: Gated Cross-Attention (Pseudocode)

```python
# Pseudocode for gated cross-attention
h_cross = gated_cross_attention(h_T, h_I)
```

## 5. Summary

Image-to-text models like BLIP, OFA, and Flamingo enable machines to describe images, answer questions, and ground language in vision. Their architectures and training objectives are key to their success. 