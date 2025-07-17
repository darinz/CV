# 07 Text-to-Image Generation

Text-to-image generation is the task of generating a realistic image conditioned on a text prompt. This is a challenging and creative application of vision-language models.

## Problem Definition
Given a text prompt $T$, generate an image $I$ that matches the description.

## Model Approaches
- **Diffusion Models:** Iteratively refine noise into an image, conditioned on text.
- **GANs:** Use a generator and discriminator, conditioned on text embedding.
- **Autoregressive Transformers:** Generate images pixel-by-pixel or patch-by-patch, conditioned on text.

### Mathematical Formulation
- $I = G(f_{text}(T), z)$, where $G$ is a generator, $f_{text}(T)$ is the text embedding, and $z$ is random noise.

## Example: Text-to-Image with Diffusion (Conceptual PyTorch Pseudocode)

```python
# Pseudocode for text-to-image generation using a diffusion model
text = "A dog riding a skateboard"
text_emb = text_encoder(text)  # e.g., CLIP text encoder
image = diffusion_model.generate(text_emb)
# image: generated image conditioned on text
```

## Example: Text-to-Image with Conditional GAN (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleTextToImageGAN(nn.Module):
    def __init__(self, text_dim, noise_dim, img_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
    def forward(self, text_emb, noise):
        x = torch.cat([text_emb, noise], dim=-1)
        return self.fc(x)

# Dummy data
text_emb = torch.randn(4, 64)
noise = torch.randn(4, 32)
model = SimpleTextToImageGAN(text_dim=64, noise_dim=32, img_dim=3*32*32)
img = model(text_emb, noise)
print('Generated image shape:', img.shape)  # (4, 3072)
```

### Explanation
- **Diffusion:** Gradually denoise random noise to create an image, guided by text embedding.
- **GAN:** Concatenate text embedding and noise, generate image with neural network.

## Real-World Models
- **DALL-E, Stable Diffusion, Imagen:** State-of-the-art text-to-image models.

## Summary
Text-to-image generation enables creative AI applications, allowing users to generate images from natural language descriptions. 