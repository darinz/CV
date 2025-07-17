# Real-Time Object Detection and Tracking

This guide covers the core concepts, mathematics, and practical Python code for building real-time object detection and tracking systems.

## 1. Real-Time Detection Pipeline

A real-time detection system processes each video frame as it arrives, aiming for minimal latency.

### Frame Processing Time
The total time to process a frame is:

```math
T_{total} = T_{preprocess} + T_{inference} + T_{postprocess} + T_{tracking}
```

- **Preprocess:** Resize, normalize, etc.
- **Inference:** Run the neural network.
- **Postprocess:** Decode outputs, apply NMS.
- **Tracking:** Update object tracks.

**Target FPS:**
```math
\text{FPS} = \frac{1}{T_{total}}
```

### Latency Budget
For a target FPS, the latency budget per frame is:
```math
T_{budget} = \frac{1000}{\text{target\_fps}} \text{ ms}
```

## 2. Real-Time Object Detection

### YOLO Real-Time Variants
YOLO (You Only Look Once) is a family of fast object detectors. For real-time, use lightweight versions like YOLOv4-tiny.

**Input:** 416x416 RGB image
**Output:** Bounding boxes, class, score for each detected object.

#### Python Example: YOLOv5 Inference
```python
import torch
from PIL import Image
import numpy as np

# Load YOLOv5 model (requires 'pip install torch torchvision yolov5')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = Image.open('sample.jpg')
results = model(img)
results.print()  # Print results
results.show()   # Show detections
```

### MobileNet-SSD
MobileNet-SSD is optimized for mobile/edge devices using depthwise separable convolutions.

#### Depthwise Separable Convolution
- **Depthwise:** Applies a single filter per input channel.
- **Pointwise:** 1x1 convolution to mix channels.

**Python Example: Depthwise Separable Conv (PyTorch)**
```python
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

## 3. Object Tracking

### Kalman Filter
A Kalman Filter predicts the next position of an object based on its previous state and updates it with new measurements.

#### State Prediction
```math
\hat{x}_t = F_t x_{t-1} + B_t u_t
```

#### Measurement Update
```math
x_t = \hat{x}_t + K_t (z_t - H_t \hat{x}_t)
```

#### Python Example: Kalman Filter (1D)
```python
import numpy as np

def kalman_1d(z, x_prev, P_prev, F=1, H=1, Q=0.01, R=1):
    # Predict
    x_pred = F * x_prev
    P_pred = F * P_prev * F + Q
    # Update
    K = P_pred * H / (H * P_pred * H + R)
    x_new = x_pred + K * (z - H * x_pred)
    P_new = (1 - K * H) * P_pred
    return x_new, P_new
```

### SORT (Simple Online and Realtime Tracking)
SORT uses a Kalman filter for each object and the Hungarian algorithm for assignment.

#### Hungarian Algorithm (Assignment)
```math
C_{ij} = 1 - \text{IoU}(b_i, b_j)
```

#### Python Example: IoU Calculation
```python
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
```

### DeepSORT
DeepSORT extends SORT by adding appearance features (ReID) for robust tracking.

#### Similarity Score
```math
\text{sim}(i, j) = \lambda \text{IoU}(b_i, b_j) + (1-\lambda) \cos(f_i, f_j)
```

- $f_i$ is the appearance feature vector for detection $i$.

## 4. Multi-Object Tracking

### Track Management
- **States:** Tentative, Confirmed, Deleted
- **Score:** Combines previous score and detection confidence

#### Track Termination
```math
\text{Delete if } \text{Score}_t < \tau_{low} \text{ for } N_{miss} \text{ frames}
```

## Summary
- Real-time detection and tracking require fast, efficient models (YOLO, MobileNet-SSD).
- Tracking uses Kalman filters and assignment algorithms (SORT, DeepSORT).
- Multi-object tracking manages track states and scores.

For more, see [PyImageSearch SORT/DeepSORT tutorials](https://pyimagesearch.com/2021/11/29/opencv-object-tracking/). 