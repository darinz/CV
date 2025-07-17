# Video Streaming Analysis

This guide explains how to analyze live video streams in real-time, focusing on buffering, adaptive processing, motion detection, background subtraction, and optimization techniques.

## 1. Streaming Pipeline

A typical streaming pipeline processes frames from a live video feed, often under strict latency constraints.

### Frame Buffer Management

A buffer temporarily stores incoming frames to smooth out variations in arrival and processing times.

**Buffer Size:**
```math
B_{size} = \text{max\_latency} \times \text{fps}
```

**Frame Drop Strategy:**
```math
\text{Drop if } \frac{B_{current}}{B_{size}} > \tau_{drop}
```

#### Python Example: Simple Frame Buffer
```python
from collections import deque

class FrameBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    def add(self, frame):
        self.buffer.append(frame)
    def get(self):
        if self.buffer:
            return self.buffer.popleft()
        return None
```

### Adaptive Processing

Adjust processing quality based on available time per frame.

**Quality Scaling:**
```math
Q_{scale} = \min(1.0, \frac{T_{budget}}{T_{current}})
```

**Resolution Scaling:**
```math
H_{new} = H_{original} \times \sqrt{Q_{scale}}
W_{new} = W_{original} \times \sqrt{Q_{scale}}
```

#### Python Example: Adaptive Resolution
```python
import cv2
import numpy as np

def adaptive_resize(frame, q_scale):
    h, w = frame.shape[:2]
    scale = np.sqrt(q_scale)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(frame, (new_w, new_h))
```

## 2. Real-Time Video Analytics

### Motion Detection

Detects movement by comparing consecutive frames.

**Frame Difference:**
```math
D_t = \|I_t - I_{t-1}\|_2
```

**Motion Score:**
```math
M_t = \frac{1}{HW} \sum_{i,j} \mathbb{1}[D_t(i,j) > \tau_{motion}]
```

#### Python Example: Simple Motion Detection
```python
import cv2
import numpy as np

def detect_motion(frame, prev_frame, threshold=25):
    diff = cv2.absdiff(frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(motion_mask > 0) / motion_mask.size
    return motion_mask, motion_score
```

### Background Subtraction

Maintains a running average of the background to detect foreground objects.

**Background Model:**
```math
B_t = \alpha B_{t-1} + (1-\alpha) I_t
```

**Foreground Detection:**
```math
F_t = \|I_t - B_t\| > \tau_{bg}
```

#### Python Example: Running Average Background Subtraction
```python
import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.bg = None
    def apply(self, frame):
        if self.bg is None:
            self.bg = frame.astype(float)
        self.bg = self.alpha * self.bg + (1 - self.alpha) * frame
        fg_mask = cv2.absdiff(frame, self.bg.astype(np.uint8))
        return fg_mask
```

## 3. Streaming Optimization

### Temporal Sampling

**Adaptive Sampling:**
```math
\text{Skip frames if } \frac{T_{processing}}{T_{budget}} > 1.0
```

**Key Frame Detection:**
```math
\text{Key frame if } \|I_t - I_{last\_key}\| > \tau_{key}
```

#### Python Example: Key Frame Detection
```python
def is_key_frame(frame, last_key_frame, threshold=50):
    diff = np.linalg.norm(frame.astype(float) - last_key_frame.astype(float))
    return diff > threshold
```

## Summary
- Video streaming analysis requires careful buffer management and adaptive processing.
- Motion detection and background subtraction are key analytics tasks.
- Optimization techniques like temporal sampling and key frame detection help maintain real-time performance. 