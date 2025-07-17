# 8. Applications of Domain Adaptation and Transfer Learning

Domain adaptation and transfer learning are widely used in real-world applications, especially in computer vision and natural language processing (NLP). This guide explains key applications and provides Python code examples.

---

## 8.1 Computer Vision Applications

### 8.1.1 Image Classification
Classify images into categories, even when training and test data come from different domains (e.g., different cameras).

**Mathematical Formulation:**
$$
f: \mathcal{X} \rightarrow \mathcal{Y}
$$

**Python Example (PyTorch):**
```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# Example usage:
model = SimpleClassifier(512, 10)
img_feat = torch.randn(1, 512)
pred = model(img_feat)
```

### 8.1.2 Object Detection
Detect and localize objects in images, adapting to new domains (e.g., different lighting).

**Mathematical Formulation:**
$$
f: \mathcal{X} \rightarrow \{(b_i, c_i)\}_{i=1}^{N}
$$

**Python Example (Pseudo-code):**
```python
# Using torchvision's Faster R-CNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
# model can be fine-tuned for domain adaptation
```

### 8.1.3 Semantic Segmentation
Assign a class label to each pixel, adapting to new domains (e.g., different sensors).

**Mathematical Formulation:**
$$
f: \mathcal{X} \rightarrow \mathcal{Y}^{H \times W}
$$

**Python Example (Pseudo-code):**
```python
# Using torchvision's DeepLabV3
from torchvision.models.segmentation import deeplabv3_resnet50
model = deeplabv3_resnet50(pretrained=True)
# model can be fine-tuned for domain adaptation
```

---

## 8.2 Natural Language Processing (NLP) Applications

### 8.2.1 Text Classification
Classify text documents, adapting to new topics or domains.

**Mathematical Formulation:**
$$
f: \mathcal{T} \rightarrow \mathcal{Y}
$$

**Python Example (HuggingFace Transformers):**
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
inputs = tokenizer("This is a test.", return_tensors="pt")
outputs = model(**inputs)
```

### 8.2.2 Named Entity Recognition (NER)
Identify entities (names, locations, etc.) in text, adapting to new domains (e.g., medical, legal).

**Mathematical Formulation:**
$$
f: \mathcal{T} \rightarrow \mathcal{Y}^L
$$

**Python Example (HuggingFace Transformers):**
```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')
inputs = tokenizer("John lives in New York.", return_tensors="pt")
outputs = model(**inputs)
```

---

## Summary
- **Image Classification, Object Detection, Semantic Segmentation**: Key computer vision tasks
- **Text Classification, NER**: Key NLP tasks

Domain adaptation and transfer learning enable robust performance across diverse, real-world environments. 