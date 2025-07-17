# Image Segmentation

Image segmentation is the process of partitioning an image into meaningful regions, such as objects or parts of objects. This guide covers semantic, instance, and panoptic segmentation, with detailed explanations, math, and Python code examples.

## 1. Overview

- **Semantic Segmentation**: Assigns a class label to each pixel (e.g., sky, road, car).
- **Instance Segmentation**: Distinguishes between different instances of the same class (e.g., separates two cars).
- **Panoptic Segmentation**: Combines semantic and instance segmentation for a complete scene understanding.

---

## 2. Semantic Segmentation

### 2.1 FCN (Fully Convolutional Networks)

FCNs replace fully connected layers with convolutional layers, enabling pixel-wise prediction.

**Key Concepts:**
- Upsampling (deconvolution) to restore spatial resolution
- Skip connections to combine low- and high-level features

**Mathematical Formulation:**
- Upsampling: $F_{up} = \text{upsample}(F, \text{scale})$
- Pixel-wise classification: $P(c|p_{ij}) = \text{softmax}(W_c F_{ij} + b_c)$

**Python Example (FCN Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load FCN model
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_t)['out'][0]

pred = output.argmax(0)
```

### 2.2 U-Net

U-Net is an encoder-decoder architecture with skip connections, popular in biomedical image segmentation.

**Key Concepts:**
- Encoder: Downsamples input to extract features
- Decoder: Upsamples to original size
- Skip connections: Concatenate encoder and decoder features

**Mathematical Formulation:**
- Encoder: $F_{enc}^l = \text{Encoder}_l(F_{enc}^{l-1})$
- Decoder: $F_{dec}^l = \text{Decoder}_l(F_{dec}^{l+1}, F_{enc}^l)$
- Skip: $F_{skip}^l = \text{Concat}(F_{dec}^l, F_{enc}^l)$

**Python Example (U-Net with Segmentation Models PyTorch):**
```python
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms

# Load U-Net model
model = smp.Unet(encoder_name="resnet34", pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    mask = model(img_t)
```

### 2.3 DeepLab

DeepLab uses atrous (dilated) convolutions and Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.

**Key Concepts:**
- Atrous convolution: $y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]$
- ASPP: Combines features at multiple dilation rates

**Python Example (DeepLabV3 Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load DeepLabV3 model
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_t)['out'][0]

pred = output.argmax(0)
```

---

## 3. Instance Segmentation

### 3.1 Mask R-CNN

Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI).

**Key Concepts:**
- RoI Align for precise region extraction
- Separate mask head for pixel-wise mask prediction

**Mathematical Formulation:**
- RoI Align: $F_{roi} = \text{RoIAlign}(F, r)$
- Mask head: $M = \text{MaskHead}(F_{roi})$

**Python Example (Mask R-CNN Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_t)[0]

masks = output['masks']  # [N, 1, H, W]
```

### 3.2 YOLACT

YOLACT is a real-time instance segmentation model that assembles masks from prototype masks and per-instance coefficients.

**Key Concepts:**
- Protonet generates prototype masks
- Prediction head outputs mask coefficients
- Final mask: $M = \sum_{k} c_k \cdot P_k$

**Python Example (YOLACT Inference):**
```python
# YOLACT requires a separate implementation, e.g., https://github.com/dbolya/yolact
# Example usage (after installing YOLACT):
# python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.3 --top_k=5 --image=image.jpg
```

---

## 4. Panoptic Segmentation

Panoptic segmentation unifies semantic and instance segmentation for a complete scene understanding.

### 4.1 Panoptic FPN

Panoptic FPN combines semantic and instance segmentation heads in a Feature Pyramid Network.

**Key Concepts:**
- Semantic branch: $S = \text{SemanticHead}(F)$
- Instance branch: $I = \text{InstanceHead}(F)$
- Fusion: $P = \text{Fusion}(S, I)$

**Python Example (Panoptic FPN Inference):**
```python
import torchvision
from PIL import Image
from torchvision import transforms

# Load Panoptic FPN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])
img = Image.open('image.jpg')
img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_t)[0]
# Panoptic segmentation requires post-processing to combine masks and semantic labels
```

---

## 5. Summary Table

| Model         | Type                | Key Feature                |
|---------------|---------------------|----------------------------|
| FCN           | Semantic            | Fully convolutional        |
| U-Net         | Semantic            | Encoder-decoder, skip conn |
| DeepLab       | Semantic            | Atrous conv, ASPP          |
| Mask R-CNN    | Instance            | RoI Align, mask head       |
| YOLACT        | Instance            | Prototype masks, fast      |
| Panoptic FPN  | Panoptic            | FPN, dual heads            |

---

## 6. References
- [FCN Paper](https://arxiv.org/abs/1411.4038)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [DeepLab Paper](https://arxiv.org/abs/1606.00915)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [YOLACT Paper](https://arxiv.org/abs/1904.02689)
- [Panoptic FPN Paper](https://arxiv.org/abs/1901.02446) 