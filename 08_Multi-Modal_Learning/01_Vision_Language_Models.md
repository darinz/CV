# Vision-Language Models

Vision-language models are designed to understand and generate content across both visual and textual modalities. They are foundational to multi-modal AI, enabling systems to connect images and text in meaningful ways. This guide covers three influential models: CLIP, DALL-E, and GPT-4V.

## 1. Introduction

Vision-language models bridge the gap between computer vision and natural language processing. They are trained on paired image-text data and can perform tasks such as image classification, zero-shot learning, image generation, and more.

## 2. CLIP (Contrastive Language-Image Pre-training)

CLIP learns visual representations by training on image-text pairs using contrastive learning. The goal is to align image and text embeddings in a shared space.

### 2.1 Architecture

- **Image Encoder:**
  - Typically a Vision Transformer (ViT) or ResNet.
  - Maps an image $`I`$ to a feature vector $`f_I(I) \in \mathbb{R}^d`$.
- **Text Encoder:**
  - Usually a Transformer.
  - Maps a text $`T`$ to a feature vector $`f_T(T) \in \mathbb{R}^d`$.

```math
\begin{align*}
f_I(I) &= \text{ViT}(I) \in \mathbb{R}^d \\
f_T(T) &= \text{Transformer}(T) \in \mathbb{R}^d
\end{align*}
```

### 2.2 Contrastive Learning

For a batch of $`N`$ image-text pairs $`(I_1, T_1), \ldots, (I_N, T_N)`$:

- Normalize embeddings:

```math
I_i = \frac{f_I(I_i)}{\|f_I(I_i)\|}, \quad T_i = \frac{f_T(T_i)}{\|f_T(T_i)\|}
```

- Compute similarity matrix:

```math
S_{ij} = I_i \cdot T_j^T
```

- Contrastive loss:

```math
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N} \exp(S_{ij}/\tau)}
```

where $`\tau`$ is a temperature parameter.

#### Python Example: CLIP-style Contrastive Loss

```python
import torch
import torch.nn.functional as F

def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    # Similarity matrix
    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(len(image_embeds)).to(image_embeds.device)
    # Cross-entropy loss (both directions)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
```

### 2.3 Zero-Shot Classification

CLIP can classify images without explicit training on the target classes by comparing image embeddings to text prompt embeddings:

```math
P(y|I) = \frac{\exp(f_I(I) \cdot f_T(T_y)/\tau)}{\sum_{c=1}^{C} \exp(f_I(I) \cdot f_T(T_c)/\tau)}
```

where $`T_y`$ is the text prompt for class $`y`$.

#### Python Example: Zero-Shot Prediction

```python
import numpy as np

def zero_shot_predict(image_embed, class_text_embeds, temperature=0.07):
    # image_embed: shape (d,)
    # class_text_embeds: shape (C, d)
    image_embed = image_embed / np.linalg.norm(image_embed)
    class_text_embeds = class_text_embeds / np.linalg.norm(class_text_embeds, axis=1, keepdims=True)
    logits = image_embed @ class_text_embeds.T / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs
```

## 3. DALL-E

DALL-E generates images from text descriptions using a discrete VAE (dVAE) and a transformer.

### 3.1 Discrete VAE

- **Encoder:** Maps image $`x`$ to latent $`z \in \mathbb{R}^{H \times W \times d}`$.
- **Quantization:** Discretizes $`z`$ to $`z_q \in \{1, \ldots, K\}^{H \times W}`$.
- **Decoder:** Reconstructs image from $`z_q`$.

```math
\begin{align*}
z &= \text{Encoder}(x) \in \mathbb{R}^{H \times W \times d} \\
z_q &= \text{Quantize}(z) \in \{1, \ldots, K\}^{H \times W} \\
\hat{x} &= \text{Decoder}(z_q)
\end{align*}
```

### 3.2 Text-to-Image Generation

The transformer models the distribution:

```math
P(z_q|T) = \prod_{i=1}^{H \times W} P(z_{q,i}|z_{q,<i}, T)
```

where $`T`$ is the text prompt.

#### Python Example: Sampling from a Transformer

```python
# Pseudocode for autoregressive sampling
z_q = []
for i in range(H * W):
    probs = transformer.predict(z_q, text_prompt)
    z_q.append(np.random.choice(K, p=probs))
# Decode z_q to image
image = decoder(z_q)
```

## 4. GPT-4V (GPT-4 Vision)

GPT-4V processes both text and images in a unified transformer architecture.

### 4.1 Multi-Modal Input

```math
X = [T_1, T_2, \ldots, T_n, I_1, I_2, \ldots, I_m]
```

where $`T_i`$ are text tokens and $`I_j`$ are image tokens.

### 4.2 Vision Encoder

```math
I_{tokens} = \text{VisionEncoder}(I) \in \mathbb{R}^{n_I \times d}
```

### 4.3 Unified Processing

```math
h = \text{Transformer}(X) \in \mathbb{R}^{(n_T + n_I) \times d}
```

#### Python Example: Unified Transformer Forward Pass

```python
# Pseudocode for unified transformer
text_tokens = text_tokenizer(text)
image_tokens = vision_encoder(image)
X = text_tokens + image_tokens
output = transformer(X)
```

## 5. Summary

Vision-language models like CLIP, DALL-E, and GPT-4V are at the core of modern multi-modal AI. They enable systems to connect vision and language, perform zero-shot learning, generate images from text, and process multi-modal inputs in a unified way. Understanding their architectures, math, and code is key to leveraging their power in real-world applications. 