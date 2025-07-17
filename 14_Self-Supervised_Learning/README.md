# Self-Supervised Learning

Self-supervised learning (SSL) is a paradigm where models learn useful representations from unlabeled data by solving auxiliary (pretext) tasks. SSL has become foundational in computer vision, natural language processing, and multimodal learning.

## Pretext Tasks

Pretext tasks are designed to generate supervisory signals from the data itself, enabling representation learning without manual labels.

### Common Pretext Tasks

- **Image Colorization:** Predict color channels from grayscale images.
- **Jigsaw Puzzle:** Predict the correct arrangement of shuffled image patches.
- **Rotation Prediction:** Predict the rotation angle applied to an image (e.g., 0°, 90°, 180°, 270°).
- **Inpainting:** Predict missing regions in an image.
- **Temporal Order Verification:** For videos, predict the correct temporal order of frames.

#### Example: Rotation Prediction
Given an image $`x`$ and a rotation $`r \in \{0, 90, 180, 270\}`$, the model predicts $`r`$:
```math
\hat{r} = f(\text{rotate}(x, r))
```
The loss is typically cross-entropy:
```math
L = -\sum_{i} y_i \log \hat{y}_i
```

## Contrastive Learning

Contrastive learning aims to learn representations by bringing similar (positive) pairs closer and pushing dissimilar (negative) pairs apart in the embedding space.

### Core Concepts
- **Positive Pair:** Two augmented views of the same instance.
- **Negative Pair:** Views from different instances.

### Contrastive Loss (InfoNCE)
Given a query $`q`$, a positive key $`k^+`$, and a set of negative keys $`\{k^-_j\}`$:
```math
L = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_j \exp(q \cdot k^-_j / \tau)}
```
where $`\tau`$ is a temperature hyperparameter.

### SimCLR Framework
- Apply random augmentations to an image to create two views $`x_i, x_j`$.
- Encode with a shared network $`f(\cdot)`$ to get representations $`h_i, h_j`$.
- Project to a lower-dimensional space $`z_i, z_j`$.
- Use contrastive loss to maximize agreement between $`z_i`$ and $`z_j`$.

### MoCo (Momentum Contrast)
- Maintains a dynamic dictionary as a queue of data samples.
- Uses a momentum encoder to update key representations.

### BYOL (Bootstrap Your Own Latent)
- Learns by predicting one view from another without negative pairs.
- Uses an online and a target network.

## Multisensory Supervision

Multisensory self-supervision leverages multiple modalities (e.g., audio, video, text) to create supervisory signals.

### Audio-Visual Correspondence
- Predict if a video frame and an audio segment are temporally aligned.

#### Example: Audio-Visual Contrastive Loss
Given visual embedding $`v`$ and audio embedding $`a`$:
```math
L = -\log \frac{\exp(v \cdot a / \tau)}{\sum_{a'} \exp(v \cdot a' / \tau)}
```

### Cross-Modal Generation
- Predict one modality from another (e.g., generate audio from video or vice versa).

### Multimodal Masked Modeling
- Mask out parts of one modality and predict them using information from another modality.

## Evaluation of Self-Supervised Representations

- **Linear Evaluation Protocol:** Train a linear classifier on frozen representations.
- **Transfer Learning:** Fine-tune on downstream tasks (e.g., classification, detection, segmentation).
- **Clustering Metrics:** Evaluate how well representations group similar instances.

## Summary

Self-supervised learning enables models to learn from vast amounts of unlabeled data by designing clever pretext tasks, leveraging contrastive objectives, and exploiting multisensory signals. These methods have led to significant advances in representation learning and have become standard practice in modern machine learning pipelines. 