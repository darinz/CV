# 03 Multimodal Generation

Multimodal generation refers to generating data in one modality conditioned on another, such as generating text from images (image captioning) or images from text (text-to-image synthesis).

## Types of Multimodal Generation
- **Image Captioning:** Generate a descriptive sentence for an image.
- **Text-to-Image Generation:** Generate an image from a text prompt.

## Mathematical Formulation
- **Image Captioning:**
  - Given image $I$, generate sentence $S = (w_1, ..., w_T)$
  - Conditional language model:
    ```math
    p(S|I) = \prod_{t=1}^T p(w_t | w_{<t}, v)
    ```
    where $v = f_{\text{CNN}}(I)$
- **Text-to-Image:**
  - Given text $T$, generate image $I$
  - $I = G(f_{text}(T), z)$, where $G$ is a generator (e.g., diffusion, GAN, transformer), $z$ is noise.

## Example 1: Image Captioning (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleImageCaptioner(nn.Module):
    def __init__(self, img_dim, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.img_fc = nn.Linear(img_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, img_feat, captions):
        # img_feat: (batch, img_dim)
        # captions: (batch, seq_len)
        h0 = self.img_fc(img_feat).unsqueeze(0)  # (1, batch, hidden_dim)
        emb = self.embedding(captions)
        out, _ = self.rnn(emb, h0)
        logits = self.fc_out(out)
        return logits

# Dummy data
img_feat = torch.randn(4, 256)
captions = torch.randint(0, 1000, (4, 10))
model = SimpleImageCaptioner(img_dim=256, vocab_size=1000, embed_dim=64, hidden_dim=128)
logits = model(img_feat, captions)
print('Logits shape:', logits.shape)  # (4, 10, 1000)
```

## Example 2: Text-to-Image (Conceptual, using Diffusion)

```python
# Pseudocode for text-to-image generation
text = "A cat sitting on a mat"
text_emb = text_encoder(text)  # e.g., CLIP text encoder
image = diffusion_model.generate(text_emb)
# image: generated image conditioned on text
```

## Explanation
- **Image Captioning:** Encodes image, uses RNN to generate caption.
- **Text-to-Image:** Encodes text, uses generative model to synthesize image.

## Real-World Models
- **BLIP, OFA:** Multimodal generation (captioning, VQA, more)
- **Stable Diffusion, DALL-E:** Text-to-image generation

## Summary
Multimodal generation enables creative and practical applications by bridging vision and language, allowing models to generate one modality from another. 