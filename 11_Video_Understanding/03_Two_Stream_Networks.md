# Two-Stream Networks

Two-stream networks process spatial and temporal information separately, then fuse them for video understanding. This guide covers the spatial stream, temporal stream (optical flow), and fusion strategies, with detailed explanations and Python code examples.

---

## 1. Spatial Stream

The spatial stream processes individual frames to capture appearance (what things look like).

$$
h_t^{spatial} = f_{spatial}(v_t) \in \mathbb{R}^d
$$

- $f_{spatial}$: a CNN (e.g., ResNet)
- $v_t$: frame $t$

**Python Example:**
```python
# Use the same frame feature extraction as in Video Classification
# See previous example for extract_frame_features(frames)
```

---

## 2. Temporal Stream (Optical Flow)

The temporal stream processes motion information, typically using optical flow between frames.

### Optical Flow

Optical flow estimates pixel-wise motion between consecutive frames.

#### Lucas-Kanade Method (Classical)

$$
\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}
$$

- $I_x, I_y, I_t$: spatial and temporal derivatives
- $u, v$: flow vectors

#### DeepFlow (Deep Learning)

$$
f = \text{DeepFlow}(v_t, v_{t+1}) = \text{CNN}([v_t, v_{t+1}])
$$

### Python Example: Compute Optical Flow (OpenCV)
```python
import cv2
import numpy as np

def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow  # shape: (H, W, 2)
```

---

## 3. Fusion Strategies

Combine spatial and temporal features for final classification.

### a) Late Fusion (Concatenation)
$$
h_{fused} = \text{Concat}(h^{spatial}, h^{temporal})
$$

### b) Weighted Fusion
$$
h_{fused} = \alpha h^{spatial} + (1-\alpha) h^{temporal}
$$

### c) Attention Fusion
$$
\alpha = \sigma(W_a [h^{spatial}; h^{temporal}] + b_a)
$$
$$
h_{fused} = \alpha h^{spatial} + (1-\alpha) h^{temporal}
$$

**Python Example (Late Fusion):**
```python
import numpy as np

def late_fusion(spatial_feat, temporal_feat):
    return np.concatenate([spatial_feat, temporal_feat])
```

---

## Summary

- Two-stream networks use separate spatial and temporal pathways
- Optical flow captures motion for the temporal stream
- Fusion strategies combine both streams for robust video classification

This approach improves performance on action recognition and other video tasks. 