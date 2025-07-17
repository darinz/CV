# Object Detection

Object detection is a fundamental task in computer vision, aiming to locate and classify objects within images. This guide covers both single-stage and two-stage detectors, explaining the concepts, mathematics, and providing Python code examples.

## 1. Overview

Object detection systems output bounding boxes and class labels for objects in an image. The main approaches are:
- **Single-stage detectors**: Predict bounding boxes and classes in one pass (e.g., YOLO, SSD, RetinaNet).
- **Two-stage detectors**: First propose regions, then classify and refine them (e.g., R-CNN, Fast R-CNN, Faster R-CNN).

---

## 2. Single-Stage Detectors

### 2.1 YOLO (You Only Look Once)

YOLO divides the image into a grid and predicts bounding boxes and class probabilities for each cell.

**Mathematical Formulation:**
- Each grid cell predicts $B$ bounding boxes and confidence scores:
  - $P(\text{object}) \in [0, 1]$
  - $(x, y, w, h) \in \mathbb{R}^4$
  - $C_1, C_2, \ldots, C_K \in [0, 1]^K$ (class probabilities)

**Loss Function:**
```math
L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]
+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 ]
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2
+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
```

**Python Example (YOLOv5 Inference):**
```python
import torch
from PIL import Image
from torchvision import transforms

# Load YOLOv5 model (requires ultralytics package)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load and preprocess image
img = Image.open('image.jpg')
results = model(img)

# Show results
results.show()
# Get bounding boxes, labels, and scores
boxes = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
```

### 2.2 SSD (Single Shot MultiBox Detector)

SSD uses multiple feature maps at different scales for detection.

**Key Concepts:**
- Multi-scale feature maps: $F_l \in \mathbb{R}^{H_l \times W_l \times C_l}$
- Default (anchor) boxes at each location
- Predicts offsets and confidences for each box

**Python Example (SSD Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load SSD model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    detections = model(img_t)[0]

print(detections['boxes'], detections['labels'], detections['scores'])
```

### 2.3 RetinaNet

RetinaNet introduces Focal Loss to address class imbalance.

**Focal Loss:**
```math
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
```
- $p_t = p$ if $y = 1$, else $p_t = 1 - p$
- $\alpha_t$ is the balancing parameter
- $\gamma$ is the focusing parameter

**Python Example (RetinaNet Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load RetinaNet model
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    detections = model(img_t)[0]

print(detections['boxes'], detections['labels'], detections['scores'])
```

---

## 3. Two-Stage Detectors

### 3.1 R-CNN (Region-based CNN)

- Uses selective search to propose regions
- Extracts features from each region using a CNN
- Classifies each region

**Mathematical Formulation:**
- Region proposal: $R = \text{SelectiveSearch}(I)$
- Feature extraction: $f_i = \text{CNN}(\text{crop}(I, r_i))$
- Classification: $P(c|f_i) = \text{softmax}(W_c f_i + b_c)$

**Python Example (Conceptual):**
```python
# Pseudocode for R-CNN (not runnable as-is)
regions = selective_search(image)
features = [cnn(crop(image, r)) for r in regions]
probs = [softmax(classifier(f)) for f in features]
```

### 3.2 Fast R-CNN

- Extracts features from the whole image once
- Uses RoI Pooling to extract region features
- Multi-task loss: $L = L_{cls} + \lambda L_{reg}$

**Python Example (Fast R-CNN Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load Fast R-CNN (Faster R-CNN with RPN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    detections = model(img_t)[0]

print(detections['boxes'], detections['labels'], detections['scores'])
```

### 3.3 Faster R-CNN

- Introduces Region Proposal Network (RPN) for fast region proposals
- Uses anchor boxes for multi-scale detection

**Mathematical Formulation:**
- RPN: $P(\text{object}) = \sigma(W_{cls} F + b_{cls})$
- Regression: $t = W_{reg} F + b_{reg}$
- Anchor boxes: $A = \{(s_i, r_j) | s_i \in S, r_j \in R\}$

**Python Example (Faster R-CNN Inference):**
```python
# (Same as Fast R-CNN example above)
```

---

## 4. Summary Table

| Model         | Type         | Speed      | Accuracy   | Key Feature           |
|---------------|--------------|------------|------------|----------------------|
| YOLO          | Single-stage | Very Fast  | Good       | Grid prediction      |
| SSD           | Single-stage | Fast       | Good       | Multi-scale features |
| RetinaNet     | Single-stage | Moderate   | High       | Focal loss           |
| R-CNN         | Two-stage    | Slow       | High       | Region proposals     |
| Fast R-CNN    | Two-stage    | Moderate   | High       | RoI Pooling          |
| Faster R-CNN  | Two-stage    | Fast       | High       | RPN                  |

---

## 5. References
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [SSD Paper](https://arxiv.org/abs/1512.02325)
- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
- [R-CNN Paper](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN Paper](https://arxiv.org/abs/1504.08083)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497) 