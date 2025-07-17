# Evaluation Metrics for Video Understanding

Evaluation metrics are essential for measuring the performance of video understanding models. This guide covers metrics for video classification, action recognition, and video retrieval, with detailed explanations, math, and Python code examples.

---

## 1. Video Classification

### a) Top-1 Accuracy

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max_c P(c|V_i) = y_i]
$$

- $N$: number of videos
- $P(c|V_i)$: predicted probability for class $c$
- $y_i$: true label

**Python Example:**
```python
def top1_accuracy(preds, labels):
    return (preds.argmax(axis=1) == labels).mean()
```

### b) Top-5 Accuracy

$$
\text{Top-5 Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i \in \text{top5}(P(c|V_i))]
$$

**Python Example:**
```python
def top5_accuracy(preds, labels):
    top5 = preds.argsort(axis=1)[:, -5:]
    return sum(label in row for row, label in zip(top5, labels)) / len(labels)
```

---

## 2. Action Recognition

### a) Mean Average Precision (mAP)

$$
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
$$

- $C$: number of classes
- $\text{AP}_c$: average precision for class $c$

### b) Temporal Intersection over Union (tIoU)

$$
\text{tIoU} = \frac{|I_1 \cap I_2|}{|I_1 \cup I_2|}
$$

- $I_1, I_2$: temporal intervals (predicted and ground truth)

**Python Example: tIoU**
```python
def temporal_iou(interval1, interval2):
    start_i = max(interval1[0], interval2[0])
    end_i = min(interval1[1], interval2[1])
    intersection = max(0, end_i - start_i)
    union = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])
    return intersection / union if union > 0 else 0
```

---

## 3. Video Retrieval

### a) Recall@K

$$
\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{rank}_i \leq K]
$$

- $\text{rank}_i$: rank of the correct item for query $i$

### b) Mean Reciprocal Rank (MRR)

$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
$$

**Python Example: MRR**
```python
def mean_reciprocal_rank(ranks):
    return sum(1.0 / r for r in ranks) / len(ranks)
```

---

## Summary

- Top-1 and Top-5 accuracy for classification
- mAP and tIoU for action recognition
- Recall@K and MRR for retrieval

These metrics help compare and improve video understanding models. 