# Diffusion Models

Diffusion models are a class of generative models that synthesize data by reversing a gradual noising process. They have recently achieved state-of-the-art results in image, audio, and video generation.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Model Architecture](#model-architecture)
4. [Training Objective](#training-objective)
5. [Implementation](#implementation)
6. [Applications](#applications)
7. [Advantages and Limitations](#advantages-and-limitations)
8. [Advanced Topics](#advanced-topics)

## Introduction

Diffusion models generate data by simulating a Markov chain that gradually adds noise to data and then learns to reverse this process to recover the original data.

## Mathematical Foundation

### Forward (Diffusion) Process
At each step $t$, noise is added to the data:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

- $\beta_t$ is a variance schedule.
- After $T$ steps, $x_T$ is nearly pure noise.

### Reverse (Denoising) Process
The model learns to reverse the noising process:

$$
p_\theta(x_{t-1} | x_t)
$$

- The model predicts either the mean or the noise at each step.

### Training Objective
The most common objective is a simplified denoising score matching loss:

$$
L = \mathbb{E}_{t, x_0, \epsilon} [\| \epsilon - \epsilon_\theta(x_t, t) \|^2]
$$

- $\epsilon$ is the noise added.
- $\epsilon_\theta$ is the model's prediction of the noise.

## Model Architecture

Most diffusion models use a U-Net architecture for images:

```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, out_channels, 3, padding=1),
        )
    def forward(self, x, t):
        h = self.encoder(x)
        out = self.decoder(h)
        return out
```

## Implementation: Denoising Diffusion Probabilistic Model (DDPM)

```python
import torch
import torch.nn.functional as F

def q_sample(x_0, t, noise, alphas_cumprod):
    # Forward process: add noise to x_0 at timestep t
    sqrt_alpha_cumprod = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[t]) ** 0.5
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

def p_losses(denoise_model, x_0, t, noise, alphas_cumprod):
    x_noisy = q_sample(x_0, t, noise, alphas_cumprod)
    predicted_noise = denoise_model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise)
```

## Applications
- Image synthesis (e.g., Stable Diffusion, Imagen)
- Audio generation
- Video generation
- Inpainting, super-resolution

## Advantages and Limitations

### Advantages
- **High Sample Quality:** Competes with or surpasses GANs.
- **Stable Training:** No adversarial game.
- **Flexible:** Can be conditioned on text, images, etc.

### Limitations
- **Slow Sampling:** Requires many steps to generate a sample.
- **Compute Intensive:** Training and sampling are expensive.

## Advanced Topics

### 1. Score-based Generative Models
- Use score matching to learn gradients of the data distribution.

### 2. Latent Diffusion Models
- Diffuse in a lower-dimensional latent space for efficiency.

### 3. Conditional Diffusion
- Condition generation on class labels, text, or images.

## Summary

Diffusion models are a powerful and flexible class of generative models. They have achieved state-of-the-art results in image, audio, and video generation, and are the foundation of models like Stable Diffusion and Imagen. 