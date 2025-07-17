# Multisensory Supervision in Self-Supervised Learning

Multisensory self-supervision leverages multiple modalities (e.g., audio, video, text) to create supervisory signals for representation learning. Below, we cover key approaches, mathematical foundations, and Python code examples.

---

## 1. Audio-Visual Correspondence

**Goal:** Predict if a video frame and an audio segment are temporally aligned.

### Explanation
- The model receives a pair (video frame, audio segment) and predicts whether they are from the same time.
- This encourages learning of cross-modal relationships.

### Mathematical Formulation
Let $v$ be the visual embedding, $a$ the audio embedding, and $y$ the label (1 if aligned, 0 otherwise):

$$
L = -[y \log \hat{y} + (1-y) \log (1-\hat{y})]
$$

### Python Example (PyTorch)
```python
import torch
import torch.nn as nn

class AudioVisualNet(nn.Module):
    def __init__(self, visual_dim, audio_dim):
        super().__init__()
        self.visual_fc = nn.Linear(visual_dim, 128)
        self.audio_fc = nn.Linear(audio_dim, 128)
        self.classifier = nn.Linear(256, 1)
    def forward(self, v, a):
        v = torch.relu(self.visual_fc(v))
        a = torch.relu(self.audio_fc(a))
        x = torch.cat([v, a], dim=1)
        out = torch.sigmoid(self.classifier(x))
        return out

# v: (batch, visual_dim), a: (batch, audio_dim), y: (batch, 1)
model = AudioVisualNet(visual_dim=512, audio_dim=128)
criterion = nn.BCELoss()
output = model(v, a)
loss = criterion(output, y)
```

---

## 2. Audio-Visual Contrastive Loss

**Goal:** Learn embeddings such that aligned audio-visual pairs are close, and misaligned pairs are far apart.

### Mathematical Formulation
Given visual embedding $v$ and audio embedding $a$:

$$
L = -\log \frac{\exp(v \cdot a / \tau)}{\sum_{a'} \exp(v \cdot a' / \tau)}
$$

### Python Example (PyTorch)
```python
import torch
import torch.nn.functional as F

def av_contrastive_loss(v, a, temperature=0.07):
    # v, a: (batch, D)
    v = F.normalize(v, dim=1)
    a = F.normalize(a, dim=1)
    logits = torch.matmul(v, a.T) / temperature
    labels = torch.arange(v.size(0)).to(v.device)
    loss = F.cross_entropy(logits, labels)
    return loss
```

---

## 3. Cross-Modal Generation

**Goal:** Predict one modality from another (e.g., generate audio from video or vice versa).

### Explanation
- The model is trained to reconstruct the target modality given the source modality.
- This encourages learning of shared and complementary information across modalities.

### Python Example (PyTorch, video-to-audio)
```python
import torch
import torch.nn as nn

class VideoToAudioNet(nn.Module):
    def __init__(self, visual_dim, audio_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(visual_dim, 256), nn.ReLU(),
            nn.Linear(256, audio_dim)
        )
    def forward(self, v):
        return self.fc(v)

# v: (batch, visual_dim), a_true: (batch, audio_dim)
model = VideoToAudioNet(visual_dim=512, audio_dim=128)
criterion = nn.MSELoss()
output = model(v)
loss = criterion(output, a_true)
```

---

## 4. Multimodal Masked Modeling

**Goal:** Mask out parts of one modality and predict them using information from another modality.

### Explanation
- Similar to masked language modeling, but across modalities (e.g., mask audio, predict from video).
- Encourages learning of cross-modal context.

### Python Example (PyTorch, simplified)
```python
import torch
import torch.nn as nn

class MultimodalMaskedNet(nn.Module):
    def __init__(self, visual_dim, audio_dim):
        super().__init__()
        self.visual_fc = nn.Linear(visual_dim, 128)
        self.audio_fc = nn.Linear(audio_dim, 128)
        self.mask_predictor = nn.Linear(256, audio_dim)
    def forward(self, v, a_masked):
        v = torch.relu(self.visual_fc(v))
        a = torch.relu(self.audio_fc(a_masked))
        x = torch.cat([v, a], dim=1)
        out = self.mask_predictor(x)
        return out

# v: (batch, visual_dim), a_masked: (batch, audio_dim), a_true: (batch, audio_dim)
model = MultimodalMaskedNet(visual_dim=512, audio_dim=128)
criterion = nn.MSELoss()
output = model(v, a_masked)
loss = criterion(output, a_true)
```

---

Multisensory self-supervision enables models to learn richer, more generalizable representations by leveraging the natural alignment and redundancy across modalities. 