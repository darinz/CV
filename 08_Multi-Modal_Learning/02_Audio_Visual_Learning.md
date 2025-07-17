# Audio-Visual Learning

Audio-visual learning combines auditory and visual information to enable AI systems to better understand the world. This guide covers key concepts such as correspondence, speech recognition, fusion, and sound localization, with detailed math and code examples.

## 1. Introduction

Audio-visual learning leverages the complementary nature of sound and vision. For example, humans use both lip movements and speech to understand language, and combine visual and audio cues to localize sound sources.

## 2. Audio-Visual Correspondence

The goal is to determine whether a given audio and video pair are semantically aligned (e.g., the sound matches the visual scene).

### 2.1 Contrastive Learning

Given audio-visual pairs $`(a, v)`$:

- **Audio Encoder:** $`f_A(a) = \text{AudioEncoder}(a) \in \mathbb{R}^d`$
- **Visual Encoder:** $`f_V(v) = \text{VisualEncoder}(v) \in \mathbb{R}^d`$

- **Similarity:**

```math
S(a, v) = \frac{f_A(a) \cdot f_V(v)}{\|f_A(a)\| \|f_V(v)\|}
```

- **Contrastive Loss:**

```math
L = -\log \frac{\exp(S(a, v)/\tau)}{\sum_{v' \in \mathcal{N}} \exp(S(a, v')/\tau)}
```

#### Python Example: Audio-Visual Contrastive Loss

```python
import torch
import torch.nn.functional as F

def av_contrastive_loss(audio_embeds, visual_embeds, temperature=0.07):
    audio_embeds = F.normalize(audio_embeds, dim=1)
    visual_embeds = F.normalize(visual_embeds, dim=1)
    logits = audio_embeds @ visual_embeds.t() / temperature
    labels = torch.arange(len(audio_embeds)).to(audio_embeds.device)
    loss_a2v = F.cross_entropy(logits, labels)
    loss_v2a = F.cross_entropy(logits.t(), labels)
    return (loss_a2v + loss_v2a) / 2
```

## 3. Audio-Visual Speech Recognition

Combines audio and visual cues (e.g., lip movements) to improve speech recognition, especially in noisy environments.

### 3.1 Lip Reading

```math
P(w|a, v) = \text{Decoder}(\text{Encoder}(a) + \text{Encoder}(v))
```

#### Python Example: Simple Fusion for Lip Reading

```python
# Pseudocode for combining audio and visual features
fused = audio_encoder(audio) + visual_encoder(video)
pred = decoder(fused)
```

### 3.2 Audio-Visual Fusion

Weighted combination of audio and visual features:

```math
h_{fusion} = \alpha \cdot h_{audio} + (1 - \alpha) \cdot h_{visual}
```

where $`\alpha`$ is learned or based on audio quality.

#### Python Example: Weighted Fusion

```python
def weighted_fusion(h_audio, h_visual, alpha):
    return alpha * h_audio + (1 - alpha) * h_visual
```

## 4. Sound Localization

Determining the spatial origin of a sound using both audio and visual cues.

### 4.1 Audio-Visual Alignment

```math
L_{alignment} = \sum_{t} \|f_A(a_t) - f_V(v_t)\|^2
```

### 4.2 Temporal Synchronization

```math
\tau^* = \arg\min_{\tau} \sum_{t} \|f_A(a_t) - f_V(v_{t+\tau})\|^2
```

#### Python Example: Temporal Alignment (Pseudocode)

```python
# Find best time lag for alignment
best_tau = None
min_loss = float('inf')
for tau in range(-max_lag, max_lag+1):
    loss = sum(np.linalg.norm(f_A[a_t] - f_V[v_t+tau])**2 for t in range(T))
    if loss < min_loss:
        min_loss = loss
        best_tau = tau
```

## 5. Summary

Audio-visual learning enables robust perception by fusing sound and vision. Techniques like contrastive learning, fusion, and alignment are key to applications such as speech recognition, event detection, and sound localization. 