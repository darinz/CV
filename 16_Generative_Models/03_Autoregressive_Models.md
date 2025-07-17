# Autoregressive Models

Autoregressive models are a class of generative models that generate data one element at a time, modeling the joint distribution as a product of conditional distributions. They are widely used in sequence modeling for text, images, and audio.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Model Architectures](#model-architectures)
4. [Training Objective](#training-objective)
5. [Implementation](#implementation)
6. [Applications](#applications)
7. [Advantages and Limitations](#advantages-and-limitations)
8. [Advanced Topics](#advanced-topics)

## Introduction

Autoregressive models generate data sequentially, predicting each element conditioned on the previous elements. This approach is powerful for modeling sequences such as text, audio, and images.

## Mathematical Foundation

Given data $x = (x_1, x_2, ..., x_T)$, the joint probability is factorized as:

$$
p(x) = \prod_{t=1}^T p(x_t | x_{<t})
$$

- $x_{<t}$ denotes all elements before $t$.
- The model learns to predict $x_t$ given the history $x_{<t}$.

## Model Architectures

### 1. PixelRNN/PixelCNN (Images)
- Model pixels sequentially (row-wise or pixel-wise).
- Use masked convolutions to ensure each pixel depends only on previous pixels.

```python
import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc, xc + (mask_type == 'B'):] = 0
        self.mask[:, :, yc + 1:] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
```

### 2. WaveNet (Audio)
- Models raw audio waveforms using dilated causal convolutions.

### 3. Transformer-based Models (Text, Images)
- Use self-attention to model dependencies.
- Examples: GPT, ImageGPT.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=20)
print(tokenizer.decode(output[0]))
```

## Training Objective

The model is trained to maximize the log-likelihood of the data:

$$
\mathcal{L} = \sum_{t=1}^T \log p(x_t | x_{<t})
$$

- For classification (e.g., next word), use cross-entropy loss.

## Implementation Example: Character-level Language Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h

# Example usage
vocab = list("abcdefghijklmnopqrstuvwxyz ")
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for i, ch in enumerate(vocab)}

model = CharRNN(len(vocab), 128)
input_seq = torch.tensor([[char2idx[c] for c in "hello"]])
logits, _ = model(input_seq)
probs = F.softmax(logits, dim=-1)
```

## Applications
- Text generation (language models)
- Image generation (PixelCNN, ImageGPT)
- Audio synthesis (WaveNet)

## Advantages and Limitations

### Advantages
- **Exact Likelihood:** Can compute exact log-likelihood.
- **Flexible:** Works for text, images, audio.
- **Simple Training:** Standard maximum likelihood.

### Limitations
- **Slow Generation:** Sequential sampling is slow.
- **Limited Global Context:** Early models struggle with long-range dependencies.

## Advanced Topics

### 1. Masked Self-Attention
- Used in Transformers for parallel training.

### 2. Sparse Transformers
- Efficiently model long sequences.

### 3. XLNet, Transformer-XL
- Address context length limitations.

## Summary

Autoregressive models are foundational for sequence modeling and generative tasks. They provide exact likelihoods and are widely used in NLP, image, and audio generation. Recent advances in self-attention and transformers have greatly improved their capabilities. 