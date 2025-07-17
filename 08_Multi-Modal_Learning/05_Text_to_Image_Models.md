# Text-to-Image Models

Text-to-image models generate images from textual descriptions. This guide covers Stable Diffusion, Imagen, and Midjourney, with detailed math and code examples.

## 1. Introduction

Text-to-image generation is a challenging task that requires understanding both language and vision. Modern models use diffusion processes, transformers, and advanced conditioning techniques.

## 2. Stable Diffusion

A diffusion-based model that generates images by iteratively denoising a random signal, conditioned on text.

### 2.1 Diffusion Process

- **Forward Process:**
```math
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
```
where $`\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s`$.

- **Reverse Process:**
```math
p_\theta(x_{t-1}|x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t))
```

### 2.2 Text Conditioning

```math
\tau_\theta(c) = \text{TextEncoder}(c)
```
```math
\epsilon_\theta(x_t, t, c) = \text{UNet}(x_t, t, \tau_\theta(c))
```

#### Python Example: Diffusion Step (Pseudocode)

```python
# Pseudocode for a single reverse diffusion step
x_t = ...  # current noisy image
c = text_encoder(text)
mu, sigma = unet(x_t, t, c)
x_prev = sample_normal(mu, sigma)
```

## 3. Imagen

Imagen uses cascaded diffusion and classifier-free guidance for high-fidelity text-to-image generation.

### 3.1 Cascaded Diffusion

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

### 3.2 Classifier-Free Guidance

```math
\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
```
where $`s`$ is the guidance scale.

#### Python Example: Classifier-Free Guidance (Pseudocode)

```python
def classifier_free_guidance(eps_uncond, eps_cond, scale):
    return eps_uncond + scale * (eps_cond - eps_uncond)
```

## 4. Midjourney

Midjourney uses a multi-scale generation process to refine images from coarse to fine.

### 4.1 Multi-Scale Generation

```math
x_{coarse} = \text{CoarseGenerator}(c)
```
```math
x_{refined} = \text{Refiner}(x_{coarse}, c)
```

#### Python Example: Multi-Scale Generation (Pseudocode)

```python
x_coarse = coarse_generator(text)
x_refined = refiner(x_coarse, text)
```

## 5. Summary

Text-to-image models like Stable Diffusion, Imagen, and Midjourney use advanced generative techniques to create high-quality images from text. Understanding their math and code is key to using and developing these models. 