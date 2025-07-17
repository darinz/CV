# Multi-Modal Learning

This module explores multi-modal learning, which combines information from different modalities (text, image, audio, video) to create more robust and comprehensive AI systems.

## Vision-Language Models

Vision-language models learn to understand and generate content across visual and textual modalities.

### CLIP (Contrastive Language-Image Pre-training)

CLIP learns visual representations by training on image-text pairs using contrastive learning.

#### Architecture

**Image Encoder:**
```math
f_I(I) = \text{ViT}(I) \in \mathbb{R}^{d}
```

**Text Encoder:**
```math
f_T(T) = \text{Transformer}(T) \in \mathbb{R}^{d}
```

#### Contrastive Learning

For a batch of $N$ image-text pairs $(I_1, T_1), (I_2, T_2), \ldots, (I_N, T_N)$:

```math
I_i = \frac{f_I(I_i)}{\|f_I(I_i)\|}, \quad T_i = \frac{f_T(T_i)}{\|f_T(T_i)\|}
```

**Similarity Matrix:**
```math
S_{ij} = I_i \cdot T_j^T
```

**Contrastive Loss:**
```math
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^{N} \exp(S_{ij}/\tau)}
```

where $\tau$ is the temperature parameter.

#### Zero-Shot Classification

```math
P(y|I) = \frac{\exp(f_I(I) \cdot f_T(T_y)/\tau)}{\sum_{c=1}^{C} \exp(f_I(I) \cdot f_T(T_c)/\tau)}
```

where $T_y$ is the text prompt for class $y$.

### DALL-E

DALL-E generates images from text descriptions using a discrete VAE and transformer.

#### Discrete VAE

**Encoder:**
```math
z = \text{Encoder}(x) \in \mathbb{R}^{H \times W \times d}
```

**Quantization:**
```math
z_q = \text{Quantize}(z) \in \{1, 2, \ldots, K\}^{H \times W}
```

**Decoder:**
```math
\hat{x} = \text{Decoder}(z_q)
```

#### Text-to-Image Generation

```math
P(z_q|T) = \prod_{i=1}^{H \times W} P(z_{q,i}|z_{q,<i}, T)
```

where $T$ is the text prompt.

### GPT-4V (GPT-4 Vision)

GPT-4V processes both text and images in a unified transformer architecture.

#### Multi-Modal Input

```math
X = [T_1, T_2, \ldots, T_n, I_1, I_2, \ldots, I_m]
```

where $T_i$ are text tokens and $I_j$ are image tokens.

#### Vision Encoder

```math
I_{tokens} = \text{VisionEncoder}(I) \in \mathbb{R}^{n_I \times d}
```

#### Unified Processing

```math
h = \text{Transformer}(X) \in \mathbb{R}^{(n_T + n_I) \times d}
```

## Audio-Visual Learning

Audio-visual learning combines auditory and visual information for better understanding.

### Audio-Visual Correspondence

#### Contrastive Learning

For audio-visual pairs $(a, v)$:

```math
f_A(a) = \text{AudioEncoder}(a) \in \mathbb{R}^{d}
```

```math
f_V(v) = \text{VisualEncoder}(v) \in \mathbb{R}^{d}
```

**Similarity:**
```math
S(a, v) = \frac{f_A(a) \cdot f_V(v)}{\|f_A(a)\| \|f_V(v)\|}
```

**Contrastive Loss:**
```math
L = -\log \frac{\exp(S(a, v)/\tau)}{\sum_{v' \in \mathcal{N}} \exp(S(a, v')/\tau)}
```

### Audio-Visual Speech Recognition

#### Lip Reading

```math
P(w|a, v) = \text{Decoder}(\text{Encoder}(a) + \text{Encoder}(v))
```

#### Audio-Visual Fusion

```math
h_{fusion} = \alpha \cdot h_{audio} + (1 - \alpha) \cdot h_{visual}
```

where $\alpha$ is learned or computed based on audio quality.

### Sound Localization

#### Audio-Visual Alignment

```math
L_{alignment} = \sum_{t} \|f_A(a_t) - f_V(v_t)\|^2
```

#### Temporal Synchronization

```math
\tau^* = \arg\min_{\tau} \sum_{t} \|f_A(a_t) - f_V(v_{t+\tau})\|^2
```

## Cross-Modal Retrieval and Generation

Cross-modal systems enable retrieval and generation across different modalities.

### Cross-Modal Retrieval

#### Image-Text Retrieval

**Image-to-Text:**
```math
\text{Retrieve}(I) = \arg\max_{T} \text{sim}(f_I(I), f_T(T))
```

**Text-to-Image:**
```math
\text{Retrieve}(T) = \arg\max_{I} \text{sim}(f_T(T), f_I(I))
```

#### Similarity Functions

**Cosine Similarity:**
```math
\text{sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
```

**Euclidean Distance:**
```math
\text{sim}(x, y) = -\|x - y\|^2
```

### Cross-Modal Generation

#### Text-to-Image Generation

**Conditional GAN:**
```math
G: \mathcal{Z} \times \mathcal{T} \rightarrow \mathcal{I}
```

```math
D: \mathcal{I} \times \mathcal{T} \rightarrow [0, 1]
```

**Diffusion Models:**
```math
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
```

```math
p_\theta(x_{t-1}|x_t, T) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, T), \Sigma_\theta(x_t, t))
```

#### Image-to-Text Generation

**Captioning Model:**
```math
P(w_t|w_{<t}, I) = \text{softmax}(W_{out} h_t + b_{out})
```

where $h_t = \text{Decoder}(h_{t-1}, [E_{w_{t-1}}, f_I(I)])$.

## Multimodal Fusion Strategies

Different strategies for combining information from multiple modalities.

### Early Fusion

Combine modalities at the input level:

```math
x_{fusion} = \text{Concat}(x_1, x_2, \ldots, x_M)
```

```math
h = \text{Encoder}(x_{fusion})
```

### Late Fusion

Combine modalities after individual processing:

```math
h_i = \text{Encoder}_i(x_i)
```

```math
h_{fusion} = \text{Fusion}(h_1, h_2, \ldots, h_M)
```

### Attention-Based Fusion

#### Cross-Modal Attention

```math
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k} \exp(e_{ik})}
```

```math
e_{ij} = \text{MLP}([h_i^{(1)}, h_j^{(2)}])
```

```math
h_i^{(1)} = h_i^{(1)} + \sum_{j} \alpha_{ij} h_j^{(2)}
```

#### Multi-Head Cross-Attention

```math
\text{CrossAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

### Hierarchical Fusion

```math
h_{local} = \text{LocalFusion}(h_1, h_2)
```

```math
h_{global} = \text{GlobalFusion}(h_{local})
```

### Gated Fusion

```math
g = \sigma(W_g [h_1, h_2] + b_g)
```

```math
h_{fusion} = g \odot h_1 + (1-g) \odot h_2
```

## Text-to-Image Models

Advanced models for generating images from textual descriptions.

### Stable Diffusion

#### Diffusion Process

**Forward Process:**
```math
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
```

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

**Reverse Process:**
```math
p_\theta(x_{t-1}|x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t))
```

#### Text Conditioning

```math
\tau_\theta(c) = \text{TextEncoder}(c)
```

```math
\epsilon_\theta(x_t, t, c) = \text{UNet}(x_t, t, \tau_\theta(c))
```

### Imagen

#### Cascaded Diffusion

```math
x_1 \sim p_\theta(x_1|c)
```

```math
x_2 \sim p_\theta(x_2|x_1, c)
```

```math
\vdots
```

```math
x_L \sim p_\theta(x_L|x_{L-1}, c)
```

#### Classifier-Free Guidance

```math
\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
```

where $s$ is the guidance scale.

### Midjourney

#### Multi-Scale Generation

```math
x_{coarse} = \text{CoarseGenerator}(c)
```

```math
x_{refined} = \text{Refiner}(x_{coarse}, c)
```

## Image-to-Text Models

Models that generate textual descriptions from images.

### BLIP (Bootstrapping Language-Image Pre-training)

#### Architecture

**Image Encoder:**
```math
h_I = \text{ViT}(I)
```

**Text Encoder:**
```math
h_T = \text{BERT}(T)
```

**Cross-Modal Encoder:**
```math
h_{cross} = \text{CrossAttention}(h_T, h_I)
```

#### Training Objectives

**Image-Text Contrastive:**
```math
L_{ITC} = -\log \frac{\exp(sim(I, T)/\tau)}{\sum_{T'} \exp(sim(I, T')/\tau)}
```

**Image-Text Matching:**
```math
L_{ITM} = -\log P(y|I, T)
```

**Language Modeling:**
```math
L_{LM} = -\sum_{t} \log P(w_t|w_{<t}, I)
```

### OFA (One-For-All)

#### Unified Architecture

```math
h = \text{Transformer}([I_{tokens}, T_{tokens}])
```

#### Multi-Task Learning

```math
L = \lambda_1 L_{caption} + \lambda_2 L_{vqa} + \lambda_3 L_{grounding}
```

### Flamingo

#### Perceiver Resampler

```math
h_I = \text{Perceiver}(I_{tokens})
```

#### Gated Cross-Attention

```math
h_{cross} = \text{GatedCrossAttention}(h_T, h_I)
```

## Evaluation Metrics

### Image Generation

#### FID (Fréchet Inception Distance)

```math
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
```

#### IS (Inception Score)

```math
\text{IS} = \exp(\mathbb{E}_{x} \text{KL}(p(y|x) \| p(y)))
```

### Text Generation

#### BLEU Score

```math
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
```

#### ROUGE Score

```math
\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}_{match}(gram_n)}{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}(gram_n)}
```

### Cross-Modal Retrieval

#### R@K (Recall at K)

```math
\text{R@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{rank}_i \leq K]
```

#### mAP (Mean Average Precision)

```math
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
```

## Applications

### Visual Question Answering (VQA)

```math
P(a|q, I) = \text{softmax}(W_{out} \text{Fusion}(f_Q(q), f_I(I)) + b_{out})
```

### Image Captioning

```math
P(w_t|w_{<t}, I) = \text{softmax}(W_{out} h_t + b_{out})
```

### Visual Grounding

```math
P(b|q, I) = \text{softmax}(W_{out} \text{Attention}(f_Q(q), f_I(I)) + b_{out})
```

where $b$ is the bounding box.

### Audio-Visual Scene Understanding

```math
P(s|a, v) = \text{softmax}(W_{out} \text{Fusion}(f_A(a), f_V(v)) + b_{out})
```

## Summary

Multi-modal learning enables AI systems to process and understand multiple types of information:

1. **Vision-Language Models**: CLIP, DALL-E, GPT-4V for text-image understanding
2. **Audio-Visual Learning**: Combining auditory and visual information
3. **Cross-Modal Retrieval**: Finding related content across modalities
4. **Multimodal Fusion**: Strategies for combining different modalities
5. **Text-to-Image Generation**: Creating images from text descriptions
6. **Image-to-Text Models**: Generating descriptions from images

These technologies are fundamental to building more comprehensive and intelligent AI systems that can understand and interact with the world through multiple senses. 