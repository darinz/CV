# 07. Training Strategies

Effective training strategies are crucial for achieving good performance with CNN architectures. This guide covers learning rate scheduling and data augmentation techniques.

## Learning Rate Scheduling

### Step Decay

Reduce the learning rate by a factor at predetermined steps:

```math
\alpha_t = \alpha_0ot \gamma^{\lfloor t / s \rfloor}
```

- $`\alpha_0`$: Initial learning rate
- $`\gamma`$: Decay factor (typically 0.1- $`s`$: Step size

### Cosine Annealing

Smooth, periodic decay:

```math
\alpha_t = \alpha_{min} + \frac[object Object]1{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t}[object Object]T}\pi))
```

- $`T`$: Total number of steps
- $`\alpha_{min}, \alpha_{max}`$: Minimum and maximum learning rates

### Python Example: Learning Rate Scheduler (PyTorch)

```python
import torch
import torch.optim as optim

# Step decay
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100eta_min=1e-6)

for epoch in range(100   # Training loop
    scheduler.step()
```

## Data Augmentation

### Training Augmentations

Common augmentations applied during training:

```math
\text{Random Crop: } I = \text{crop}(I, 224 \times 224```

```math
\text{Horizontal Flip: } I= \text{flip}(I, p=0.5```

```math
\text{Color Jittering: } I = I \odot \text{color\_transform}
```

### Test Augmentations

For inference, use deterministic augmentations:

```math
\text{Center Crop: } I = \text{center\_crop}(I, 224 \times 224```

```math
\text{10-Crop: } I = \text{ensemble}(\text{10})
```

### Python Example: Data Augmentation (PyTorch)

```python
import torchvision.transforms as transforms

# Training transforms
train_transform = transforms.Compose( transforms.RandomResizedCrop(224,
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4st=0.4aturation=0.41
    transforms.ToTensor(),
    transforms.Normalize(mean=[00.48504560.406 std=[00.229.2240.225
# Test transforms
test_transform = transforms.Compose([
    transforms.Resize(256,
    transforms.CenterCrop(224
    transforms.ToTensor(),
    transforms.Normalize(mean=[00.48504560.406 std=[00.2290.224, 0.225])
```

## Summary
- Learning rate scheduling helps models converge to better solutions
- Data augmentation improves generalization and robustness
- Proper training strategies are essential for achieving state-of-the-art performance

---

**Next:** [Performance Comparison](08_Performance_Comparison.md) 