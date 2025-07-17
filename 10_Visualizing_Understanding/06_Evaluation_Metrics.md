# Evaluation Metrics

Evaluation metrics are essential for assessing the performance of computer vision models. This guide covers metrics for object detection, segmentation, and adversarial robustness, with detailed explanations, math, and Python code examples.

##1erview

- **Object Detection Metrics**: mAP, IoU for bounding box evaluation
- **Segmentation Metrics**: Pixel accuracy, mean IoU, Dice coefficient
- **Adversarial Robustness**: Robust accuracy, attack success rate

---

## 2. Object Detection Metrics

### 2.1 IoU (Intersection over Union)

IoU measures the overlap between predicted and ground truth bounding boxes:

```math
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
```

**Python Example (IoU Calculation):**
```python
import numpy as np

def calculate_iou(box1, box2):
    # box format: [x1 y1, x2, y2]
    x1 = max(box10], box20)
    y1 = max(box11], box21)
    x2 = min(box12], box22)
    y2 = min(box1[3 box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box12 - box1) * (box13 box11)
    area2 = (box22 - box2) * (box23box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Example usage
box1 = [10010, 200, 200]
box2 = [1501500
iou = calculate_iou(box1, box2)
print(fIoU: {iou:.3}")
```

### 2.2 mAP (Mean Average Precision)

mAP is the mean of average precision across all classes:

```math
\text{AP} = \int_0^1 P(r) dr
```
```math
\text[object Object]mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
```

**Python Example (mAP Calculation):**
```python
def calculate_ap(recalls, precisions):
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # Interpolate precision
    for i in range(len(precisions) - 2, -1,-1        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP
    ap = 0
    for i in range(len(recalls) -1):
        ap += (recalls[i + 1recalls[i]) * precisions[i]
    
    return ap

# Example usage (simplified)
recalls = np.array(0.0, 0.2, 0.4, 0.6, 1])
precisions = np.array(1.0, 0.8, 0.6, 040])
ap = calculate_ap(recalls, precisions)
print(f"AP: {ap:.3f})
```

---## 3. Segmentation Metrics

### 31el Accuracy

Pixel accuracy measures the percentage of correctly classified pixels:

```math
\text{Accuracy} = \frac{\sum_{i=1}^[object Object]n} p_[object Object]ii}}{\sum_[object Object]i=1}^{k} \sum_{j=1}^{k} p_{ij}}
```

**Python Example (Pixel Accuracy):**
```python
def pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = pred.numel()
    return correct / total

# Example usage
pred = torch.randint(0, 10, (100et = torch.randint(0, 10 (100, 100c = pixel_accuracy(pred, target)
print(fPixel Accuracy: {acc:.3
```

### 3.2 Mean IoU

Mean IoU is the average IoU across all classes:

```math
\text[object Object]mIoU} = \frac{1}{k} \sum_{i=1k} \frac{p_[object Object]ii}}{\sum_{j=1}^{k} p_{ij} + \sum_{j=1}^{k} p_{ji} - p_{ii}}
```

**Python Example (Mean IoU):**
```python
def mean_iou(pred, target, num_classes):
    ious =  for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        iou = intersection / union if union > 0lse 0
        ious.append(iou)
    return np.mean(ious)

# Example usage
pred = torch.randint(0(100et = torch.randint(05, (100,10))
miou = mean_iou(pred, target, num_classes=5)
print(fMean IoU: {miou:.3
```

### 3.3Dice Coefficient

Dice coefficient measures the overlap between predicted and ground truth masks:

```math
\text{Dice} = \frac{2X \cap Y|}{|X| + |Y|}
```

**Python Example (Dice Coefficient):**
```python
def dice_coefficient(pred, target):
    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()
    return 2tersection / union if union > 0 else 0

# Example usage
pred = torch.randint(0, 2, (10010, dtype=torch.bool)
target = torch.randint(0, 2, (10010, dtype=torch.bool)
dice = dice_coefficient(pred, target)
print(fDice Coefficient: {dice:.3f})```

---

## 4. Adversarial Robustness Metrics

### 40.1 Robust Accuracy

Robust accuracy measures the percentage of correctly classified adversarial examples:

```math
\text{Robust Accuracy} = \frac{1}{N} \sum_[object Object]i=1}^{N} \mathbb{1}[f(x_i^{adv}) = y_i]
```

**Python Example (Robust Accuracy):**
```python
def robust_accuracy(model, dataloader, attack_fn):
    correct = 0
    total =0   for x, y in dataloader:
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            pred = model(x_adv).argmax(1        correct += (pred == y).sum()
        total += y.size(0)
    return correct / total

# Example usage (requires attack_fn implementation)
# robust_acc = robust_accuracy(model, test_loader, pgd_attack)
```

### 4.2ttack Success Rate

Attack success rate measures the percentage of successful adversarial attacks:

```math
\text[object Object]ASR} = \frac{1}{N} \sum_[object Object]i=1}^{N} \mathbb{1}[f(x_i^{adv}) \neq f(x_i)]
```

**Python Example (Attack Success Rate):**
```python
def attack_success_rate(model, dataloader, attack_fn):
    successful = 0
    total =0   for x, y in dataloader:
        x_adv = attack_fn(model, x, y)
        with torch.no_grad():
            pred_clean = model(x).argmax(1)
            pred_adv = model(x_adv).argmax(1)
        successful += (pred_adv != pred_clean).sum()
        total += y.size(0)
    return successful / total

# Example usage (requires attack_fn implementation)
# asr = attack_success_rate(model, test_loader, pgd_attack)
```

---

## 5. Summary Table

| Metric | Use Case | Range | Interpretation |
|--------|----------|-------|----------------|
| IoU | Object Detection | 0,1| Higher is better |
| mAP | Object Detection | 0,1| Higher is better |
| Pixel Accuracy | Segmentation | 0,1| Higher is better |
| Mean IoU | Segmentation | 0,1| Higher is better |
| Dice | Segmentation | 0,1| Higher is better |
| Robust Accuracy | Adversarial | 0,1| Higher is better |
| Attack Success Rate | Adversarial |0,1 | Lower is better |

---

##6. References
- [COCO Evaluation](https://cocodataset.org/#detection-eval)
- [Pascal VOC Evaluation](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Adversarial Robustness Paper](https://arxiv.org/abs/1706.06083) 