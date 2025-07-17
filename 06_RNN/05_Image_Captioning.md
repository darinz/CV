# Image Captioning with CNN-RNN

Image captioning is the task of generating natural language descriptions for images. This is typically achieved by combining Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) for sequence generation.

## Architecture Overview

The standard architecture consists of:
- **Encoder (CNN):** Extracts a feature vector from the image.
- **Decoder (RNN):** Generates a caption word-by-word, conditioned on the image features.

## Encoder (CNN)

A CNN processes the input image $`I`$ and outputs a feature vector $`v`$:

```math
v = \text{CNN}(I)
```

Where $`v \in \mathbb{R}^{d_v}`$ is the image feature vector.

## Decoder (RNN)

The RNN generates the caption sequence $`w_1, w_2, \ldots, w_T`$:

```math
h_t = \text{RNN}(h_{t-1}, [x_t, v])
```

```math
P(w_t | w_1, \ldots, w_{t-1}, v) = \text{softmax}(W_{out} h_t + b_{out})
```

Where $`x_t`$ is the embedding of the previous word.

## Attention Mechanism

Attention allows the model to focus on different parts of the image at each time step.

### Spatial Attention

Given spatial features $`V \in \mathbb{R}^{K \times d_v}`$:

```math
\alpha_t = \text{softmax}(W_a \tanh(W_h h_{t-1} + W_v V + b_a))
```

```math
v_t = \sum_{i=1}^{K} \alpha_{t,i} V_i
```

Where $`v_t`$ is the attended image feature at time $`t`$.

## Loss Function

The loss for a caption sequence is:

```math
L = -\sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1}, v)
```

## Training Strategies

### Teacher Forcing

During training, the ground truth word is used as input at each time step:

```math
h_t = \text{RNN}(h_{t-1}, [E_{w_t^*}, v])
```

Where $`w_t^*`$ is the ground truth word.

### Scheduled Sampling

Mixes ground truth and predicted words as input:

```math
w_t^{input} = \begin{cases}
w_t^* & \text{with probability } p \\
\hat{w}_t & \text{with probability } 1-p
\end{cases}
```

## Python Example: Image Captioning (Simplified)

Below is a simplified example using random image features and a basic RNN decoder:

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

class SimpleCaptionDecoder:
    def __init__(self, vocab_size, embed_size, hidden_size, img_feat_size):
        self.E = np.random.randn(vocab_size, embed_size) * 0.01
        self.W_xh = np.random.randn(hidden_size, embed_size + img_feat_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((vocab_size, 1))

    def step(self, x_t, v, h_prev):
        inp = np.vstack((x_t, v))
        h = np.tanh(np.dot(self.W_xh, inp) + np.dot(self.W_hh, h_prev) + self.b_h)
        y = np.dot(self.W_hy, h) + self.b_y
        p = softmax(y)
        return h, p

    def generate(self, v, start_idx, max_len=10):
        h = np.zeros((self.W_hh.shape[0], 1))
        idx = start_idx
        caption = [idx]
        for _ in range(max_len):
            x_t = self.E[idx].reshape(-1, 1)
            h, p = self.step(x_t, v, h)
            idx = int(np.argmax(p))
            caption.append(idx)
            if idx == 0:  # Assume 0 is <EOS>
                break
        return caption

# Example usage:
vocab = {'<EOS>':0, 'a':1, 'cat':2, 'on':3, 'mat':4}
img_feat = np.random.randn(8, 1)  # Random image feature
model = SimpleCaptionDecoder(vocab_size=5, embed_size=4, hidden_size=6, img_feat_size=8)
caption_indices = model.generate(img_feat, start_idx=1)
print('Generated caption indices:', caption_indices)
```

**Explanation:**
- The encoder (CNN) is simulated by a random image feature vector.
- The decoder generates a sequence of word indices, starting from a given word.
- In practice, the encoder would be a pre-trained CNN (e.g., ResNet), and the decoder would be trained on image-caption pairs.

---

Next: [Sequence-to-Sequence Models](06_Seq2Seq.md) 