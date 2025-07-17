# Evaluation of Self-Supervised Representations

Evaluating the quality of representations learned by self-supervised models is crucial. Here, we cover common evaluation protocols with explanations and Python code examples.

---

## 1. Linear Evaluation Protocol

**Goal:** Assess the quality of learned representations by training a linear classifier on top of frozen features.

### Explanation
- Freeze the backbone (feature extractor) trained with SSL.
- Train a simple linear classifier (e.g., logistic regression) on top using labeled data.
- High accuracy indicates good representations.

### Python Example (PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume backbone is a pretrained feature extractor
class LinearEvalNet(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        out = self.classifier(features)
        return out

# Training loop (simplified)
# for x, y in dataloader:
#     logits = model(x)
#     loss = criterion(logits, y)
#     ...
```

---

## 2. Transfer Learning

**Goal:** Fine-tune the SSL model on a downstream task (e.g., classification, detection, segmentation).

### Explanation
- Initialize the model with SSL-pretrained weights.
- Fine-tune all (or some) layers on the new task with labeled data.
- Compare performance to models trained from scratch.

### Python Example (PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume ssl_model is a pretrained model
ssl_model = ...
num_classes = 10
ssl_model.classifier = nn.Linear(ssl_model.classifier.in_features, num_classes)
optimizer = optim.Adam(ssl_model.parameters(), lr=1e-4)

# Training loop (simplified)
# for x, y in dataloader:
#     logits = ssl_model(x)
#     loss = criterion(logits, y)
#     ...
```

---

## 3. Clustering Metrics

**Goal:** Evaluate how well representations group similar instances (e.g., using k-means clustering).

### Explanation
- Extract features for all samples using the SSL model.
- Cluster the features (e.g., k-means).
- Compare clusters to ground-truth labels using metrics like Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI).

### Python Example (scikit-learn)
```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# features: (num_samples, feature_dim), labels: (num_samples,)
features = ...
labels = ...
num_clusters = len(set(labels))
kmeans = KMeans(n_clusters=num_clusters).fit(features)
preds = kmeans.labels_
ari = adjusted_rand_score(labels, preds)
nmi = normalized_mutual_info_score(labels, preds)
print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
```

---

These evaluation protocols help determine the usefulness and transferability of self-supervised representations for downstream tasks. 