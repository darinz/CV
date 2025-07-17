# Training Strategies for CNNs

Training convolutional neural networks (CNNs) effectively requires more than just defining the architecture. The following strategies are commonly used to improve convergence, generalization, and performance.

---

## 1. Data Augmentation

**Purpose:**  
Increase the diversity of the training data by applying random transformations, helping prevent overfitting.

**Common Techniques:**
- Random cropping
- Horizontal/vertical flipping
- Rotation
- Color jittering

**Example (PyTorch):**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
```

---

## 2. Learning Rate Scheduling

**Purpose:**  
Adjust the learning rate during training to improve convergence.

**Common Schedules:**
- Step decay
- Exponential decay
- Reduce on plateau

**Example (PyTorch):**
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()
```

---

## 3. Regularization

**Purpose:**  
Prevent overfitting and improve generalization.

**Techniques:**
- **L2 Regularization (Weight Decay):** Adds a penalty to large weights.
- **Dropout:** Randomly sets a fraction of activations to zero during training.

**Example (Dropout in PyTorch):**
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(16*6*6, 10)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

---

## 4. Batch Normalization

**Purpose:**  
Stabilize and accelerate training by normalizing layer inputs.

**How it works:**  
For each mini-batch, normalize the activations to have zero mean and unit variance, then scale and shift with learnable parameters.

**Mathematical Formulation:**
```math
\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
```
where $\mu_B$ and $\sigma_B^2$ are the mean and variance of the batch.

---

## 5. Early Stopping

**Purpose:**  
Stop training when validation performance stops improving, to avoid overfitting.

**Implementation:**
- Monitor validation loss/accuracy.
- Stop if no improvement for a set number of epochs.

---

## 6. Transfer Learning

**Purpose:**  
Leverage pre-trained models on large datasets (like ImageNet) and fine-tune on your own data.

**Example:**
```python
from torchvision import models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

model.fc = nn.Linear(model.fc.in_features, 10)  # Replace final layer
```

---

## 7. Mixed Precision Training

**Purpose:**  
Use lower-precision (e.g., float16) arithmetic to speed up training and reduce memory usage.

**Example (PyTorch):**
```python
scaler = torch.cuda.amp.GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 8. Hyperparameter Tuning

**Purpose:**  
Systematically search for the best values for learning rate, batch size, optimizer, etc.

**Methods:**
- Grid search
- Random search
- Bayesian optimization

---

## Summary Table

| Strategy                | Purpose                        | Example Tool/Method      |
|-------------------------|--------------------------------|--------------------------|
| Data Augmentation       | Prevent overfitting            | torchvision.transforms   |
| Learning Rate Scheduling| Improve convergence            | StepLR, ReduceLROnPlateau|
| Regularization          | Prevent overfitting            | Dropout, L2              |
| Batch Normalization     | Stabilize training             | nn.BatchNorm2d           |
| Early Stopping          | Avoid overfitting              | Custom callback          |
| Transfer Learning       | Leverage pre-trained models    | torchvision.models       |
| Mixed Precision         | Speed up, save memory          | torch.cuda.amp           |
| Hyperparameter Tuning   | Optimize performance           | Optuna, Ray Tune         |
