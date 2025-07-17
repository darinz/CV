# 04 Image Captioning

Image captioning is the task of generating a descriptive sentence for a given image. It combines computer vision (to understand the image) and natural language processing (to generate text).

## Problem Definition
Given an image $I$, generate a sentence $S = (w_1, ..., w_T)$ describing the image.

## Encoder-Decoder Framework
- **Encoder:** Extracts image features (e.g., using a CNN or Vision Transformer).
- **Decoder:** Generates a sentence conditioned on the image features (e.g., using an RNN or Transformer).

### Mathematical Formulation
- $v = f_{\text{CNN}}(I)$: Image feature vector
- $p(S|I) = \prod_{t=1}^T p(w_t | w_{<t}, v)$: Probability of sentence given image

## Example: Image Captioning with PyTorch

Below is a simplified encoder-decoder model for image captioning.

```python
import torch
import torch.nn as nn

class SimpleImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc = nn.Linear(2048, output_dim)  # Assume 2048-dim image features
    def forward(self, x):
        return self.fc(x)

class SimpleCaptionDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, captions, img_feat):
        emb = self.embedding(captions)
        # Use image feature as initial hidden state
        h0 = img_feat.unsqueeze(0)
        out, _ = self.rnn(emb, h0)
        logits = self.fc_out(out)
        return logits

# Dummy data
img_feat = torch.randn(2, 2048)  # 2 images
captions = torch.randint(0, 1000, (2, 12))  # 2 captions, 12 tokens each
encoder = SimpleImageEncoder(output_dim=128)
decoder = SimpleCaptionDecoder(embed_dim=64, hidden_dim=128, vocab_size=1000)

img_emb = encoder(img_feat)
logits = decoder(captions, img_emb)
print('Logits shape:', logits.shape)  # (2, 12, 1000)
```

### Explanation
- **Encoder:** Projects image features to embedding.
- **Decoder:** Generates caption, conditioned on image embedding as initial hidden state.
- **Logits:** Predicts next word at each time step.

## Training
- **Loss:** Cross-entropy between predicted and ground-truth words.
- **Teacher forcing:** Use ground-truth words as input during training.

## Real-World Models
- **Show and Tell, Show, Attend and Tell:** Classic encoder-decoder models.
- **BLIP, OFA:** Modern transformer-based models.

## Summary
Image captioning bridges vision and language, enabling machines to describe visual content in natural language. 