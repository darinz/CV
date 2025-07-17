# Applications

Computer vision techniques are widely used in real-world applications. This guide covers three major domains: autonomous driving, medical imaging, and security, with detailed explanations and Python code examples.

## 1. Overview

- **Autonomous Driving**: Uses object detection and semantic segmentation for safe navigation.
- **Medical Imaging**: Uses deep learning for tumor detection and organ segmentation.
- **Security**: Uses adversarial defense and anomaly detection to improve robustness.

---

## 2. Autonomous Driving

### 2.1 Object Detection

Object detection is crucial for identifying pedestrians, vehicles, and traffic signs in real time.

**Mathematical Formulation:**
```math
f: \mathcal{X} \rightarrow \{(b_i, c_i, s_i)\}_{i=1}^{N}
```
where $b_i$ is the bounding box, $c_i$ is the class, and $s_i$ is the confidence score.

**Python Example (YOLOv5 for Autonomous Driving):**
```python
import torch
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = Image.open('road_image.jpg')
results = model(img)
results.show()  # Visualize detections
# Get bounding boxes, labels, and scores
for *box, conf, cls in results.xyxy[0]:
    print(f"Box: {box}, Confidence: {conf:.2f}, Class: {results.names[int(cls)]}")
```

### 2.2 Semantic Segmentation

Semantic segmentation provides pixel-level understanding of the scene (e.g., road, sidewalk, car).

**Mathematical Formulation:**
```math
f: \mathcal{X} \rightarrow \mathcal{Y}^{H \times W}
```

**Python Example (DeepLabV3 for Road Segmentation):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])
img = Image.open('road_image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_t)['out'][0]
pred = output.argmax(0)
# pred contains the class index for each pixel
```

---

## 3. Medical Imaging

### 3.1 Tumor Detection

Tumor detection involves classifying medical images for the presence of tumors.

**Mathematical Formulation:**
```math
P(\text{tumor}|x) = \sigma(f_\theta(x))
```

**Python Example (Binary Classification with ResNet):**
```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Binary classification
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = Image.open('mri_scan.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    prob = torch.sigmoid(model(img_t)).item()
    print(f"Tumor probability: {prob:.2f}")
```

### 3.2 Organ Segmentation

Organ segmentation involves pixel-wise classification of medical images (e.g., segmenting the liver in a CT scan).

**Mathematical Formulation:**
```math
f: \mathcal{X} \rightarrow \{0, 1\}^{H \times W}
```

**Python Example (U-Net for Organ Segmentation):**
```python
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms

model = smp.Unet(encoder_name="resnet34", pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
img = Image.open('ct_scan.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    mask = model(img_t)
    organ_mask = (mask > 0.5).float()
```

---

## 4. Security

### 4.1 Adversarial Defense

Adversarial defense involves making models robust against adversarial attacks.

**Mathematical Formulation:**
```math
f_{robust}: \mathcal{X} \rightarrow \mathcal{Y}
```

**Python Example (Adversarial Training Step):**
```python
import torch
import torch.nn as nn

def adversarial_training_step(model, x, y, optimizer, epsilon=0.03):
    x.requires_grad_()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    # Train on both clean and adversarial examples
    output_clean = model(x.detach())
    output_adv = model(x_adv.detach())
    loss_clean = nn.CrossEntropyLoss()(output_clean, y)
    loss_adv = nn.CrossEntropyLoss()(output_adv, y)
    total_loss = loss_clean + loss_adv
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss
```

### 4.2 Anomaly Detection

Anomaly detection identifies unusual patterns in data (e.g., fraud, defects).

**Mathematical Formulation:**
```math
P(\text{anomaly}|x) = 1 - P(\text{normal}|x)
```

**Python Example (Autoencoder for Anomaly Detection):**
```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def detect_anomaly(model, x, threshold=0.1):
    with torch.no_grad():
        reconstructed = model(x)
        mse = torch.mean((x - reconstructed) ** 2)
        is_anomaly = mse > threshold
    return is_anomaly, mse.item()

# Example usage
# model = Autoencoder()
# anomaly, mse = detect_anomaly(model, test_image)
```

---

## 5. Summary Table

| Application         | Technique             | Key Challenge         | Solution                |
|---------------------|----------------------|----------------------|-------------------------|
| Autonomous Driving  | Object Detection     | Real-time processing | YOLO, SSD               |
| Autonomous Driving  | Semantic Segmentation| Scene understanding  | DeepLab, U-Net          |
| Medical Imaging     | Tumor Detection      | Limited data         | Transfer learning       |
| Medical Imaging     | Organ Segmentation   | Precision required   | U-Net, attention        |
| Security            | Adversarial Defense  | Robustness           | Adversarial training    |
| Security            | Anomaly Detection    | Unsupervised learning| Autoencoders            |

---

## 6. References
- [YOLO for Autonomous Driving](https://arxiv.org/abs/1506.02640)
- [DeepLab for Road Segmentation](https://arxiv.org/abs/1606.00915)
- [U-Net for Medical Imaging](https://arxiv.org/abs/1505.04597)
- [Adversarial Training](https://arxiv.org/abs/1706.06083)
- [Anomaly Detection Survey](https://arxiv.org/abs/1901.03407) 