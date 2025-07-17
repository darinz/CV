# Video Classification

Video classification is the task of assigning a label to a video sequence, considering both spatial (appearance) and temporal (motion) information. This guide covers the problem formulation, temporal modeling, and video-level classification, with detailed explanations and Python code examples.

---

## Problem Formulation

Given a video sequence $V = \{v_1, v_2, \ldots, v_T\}$, where each frame $v_t \in \mathbb{R}^{H \times W \times C}$ (height, width, channels):

$$
f: \mathcal{V} \rightarrow \mathcal{Y}
$$

- $\mathcal{V}$: the space of all possible videos
- $\mathcal{Y}$: the set of possible classes (e.g., 'running', 'jumping', 'cooking')

**Goal:** Learn a function $f$ that maps a video $V$ to its class label $y$.

---

## Temporal Modeling

Videos have a temporal dimension (time), so we need to model how information evolves across frames.

### 1. Frame-level Feature Extraction

Extract features from each frame using a CNN (e.g., ResNet, MobileNet):

$$
h_t = f_\theta(v_t) \in \mathbb{R}^d
$$

- $f_\theta$: a neural network (e.g., a pretrained CNN)
- $h_t$: feature vector for frame $t$

**Python Example:**
```python
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Load a pretrained ResNet
resnet = models.resnet18(pretrained=True)
resnet.eval()

# Remove the final classification layer to get features
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Preprocessing for input images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_frame_features(frames):
    features = []
    for frame in frames:
        img = transform(frame).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            feat = feature_extractor(img).squeeze().numpy()
        features.append(feat)
    return features

# Example usage:
# frames = [Image.open(path) for path in video_frame_paths]
# frame_features = extract_frame_features(frames)
```

### 2. Temporal Aggregation

Aggregate frame-level features into a single video-level feature vector.

#### a) Mean Pooling
$$
h_{video} = \frac{1}{T} \sum_{t=1}^{T} h_t
$$

#### b) Max Pooling
$$
h_{video} = \max_{t=1,\ldots,T} h_t
$$

#### c) Attention-based Aggregation
Learn weights $\alpha_t$ for each frame:
$$
\alpha_t = \frac{\exp(w^T h_t)}{\sum_{t'=1}^{T} \exp(w^T h_{t'})}
$$
$$
h_{video} = \sum_{t=1}^{T} \alpha_t h_t
$$

**Python Example (Mean Pooling):**
```python
import numpy as np

def mean_pooling(features):
    return np.mean(features, axis=0)

# video_feature = mean_pooling(frame_features)
```

**Python Example (Attention):**
```python
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.Linear(feature_dim, 1)
    def forward(self, features):
        # features: (T, d)
        attn_weights = F.softmax(self.attn(features), dim=0)  # (T, 1)
        weighted = features * attn_weights  # (T, d)
        return weighted.sum(dim=0)  # (d,)

# Example usage:
# features = torch.tensor(frame_features, dtype=torch.float32)
# aggregator = AttentionAggregator(features.shape[1])
# video_feature = aggregator(features)
```

---

## Video-level Classification

Once we have a video-level feature $h_{video}$, we can classify the video:

$$
P(y|V) = \text{softmax}(W_{out} h_{video} + b_{out})
$$

- $W_{out}$, $b_{out}$: learnable parameters

**Python Example:**
```python
class VideoClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, video_feature):
        return F.softmax(self.fc(video_feature), dim=-1)

# Example usage:
# classifier = VideoClassifier(feature_dim=512, num_classes=10)
# probs = classifier(video_feature)
```

---

## Loss Function

For training, use cross-entropy loss:

$$
L = -\sum_{i=1}^{N} \log P(y_i|V_i)
$$

- $N$: number of training videos
- $y_i$: true label for video $i$

**Python Example:**
```python
criterion = nn.CrossEntropyLoss()
# logits = classifier(video_feature.unsqueeze(0))  # Add batch dimension
# loss = criterion(logits, torch.tensor([label]))
```

---

## Summary

- Extract frame-level features using a CNN
- Aggregate features temporally (mean, max, attention)
- Classify the video using a fully connected layer
- Train with cross-entropy loss

This approach forms the basis for more advanced video understanding models, such as 3D CNNs and two-stream networks. 