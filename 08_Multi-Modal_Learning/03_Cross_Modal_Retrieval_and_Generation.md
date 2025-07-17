# Cross-Modal Retrieval and Generation

Cross-modal systems enable retrieval and generation across different modalities, such as finding images from text or generating captions for images. This guide covers retrieval, similarity functions, GANs, diffusion models, and captioning.

## 1. Introduction

Cross-modal learning allows AI to connect and translate between different types of data (e.g., text, images, audio). Common tasks include image-text retrieval and cross-modal generation.

## 2. Cross-Modal Retrieval

### 2.1 Image-Text Retrieval

- **Image-to-Text:** Retrieve the most relevant text for a given image.
- **Text-to-Image:** Retrieve the most relevant image for a given text.

```math
\text{Retrieve}(I) = \arg\max_{T} \text{sim}(f_I(I), f_T(T))
```
```math
\text{Retrieve}(T) = \arg\max_{I} \text{sim}(f_T(T), f_I(I))
```

### 2.2 Similarity Functions

- **Cosine Similarity:**
```math
\text{sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
```
- **Euclidean Distance:**
```math
\text{sim}(x, y) = -\|x - y\|^2
```

#### Python Example: Cosine Similarity

```python
import numpy as np

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

## 3. Cross-Modal Generation

### 3.1 Text-to-Image Generation

- **Conditional GAN:**
```math
G: \mathcal{Z} \times \mathcal{T} \rightarrow \mathcal{I}
```
```math
D: \mathcal{I} \times \mathcal{T} \rightarrow [0, 1]
```

- **Diffusion Models:**
```math
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
```
```math
p_\theta(x_{t-1}|x_t, T) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, T), \Sigma_\theta(x_t, t))
```

#### Python Example: GAN Forward Pass (Pseudocode)

```python
z = np.random.randn(latent_dim)
image = generator(z, text_embedding)
real_or_fake = discriminator(image, text_embedding)
```

### 3.2 Image-to-Text Generation (Captioning)

- **Captioning Model:**
```math
P(w_t|w_{<t}, I) = \text{softmax}(W_{out} h_t + b_{out})
```
where $`h_t = \text{Decoder}(h_{t-1}, [E_{w_{t-1}}, f_I(I)])`$.

#### Python Example: Image Captioning (Pseudocode)

```python
# Pseudocode for image captioning
features = image_encoder(image)
caption = []
h = decoder.init_state()
for t in range(max_len):
    h = decoder.step(h, features, prev_word)
    word = output_layer(h)
    caption.append(word)
    if word == '<EOS>':
        break
```

## 4. Summary

Cross-modal retrieval and generation are essential for tasks like search, captioning, and generative art. They rely on embedding alignment, similarity functions, and generative models such as GANs and diffusion models. 