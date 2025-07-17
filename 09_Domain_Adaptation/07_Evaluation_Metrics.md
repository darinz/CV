# 7. Evaluation Metrics for Domain Adaptation and Transfer Learning

Evaluating models in domain adaptation, few-shot, and zero-shot learning requires specialized metrics. This guide explains the most important metrics, their math, and provides Python code examples.

---

## 7.1 Domain Adaptation Metrics

### Target Accuracy
Measures the accuracy of the model on the target domain:
$$
\text{Accuracy} = \frac{1}{|\mathcal{D}_t|} \sum_{(x,y) \in \mathcal{D}_t} \mathbb{1}[\arg\max_c P(c|x) = y]
$$

**Python Example:**
```python
def accuracy(preds, labels):
    return (preds == labels).mean()
```

### H-score
H-score combines source and target accuracy:
$$
\text{H-score} = 2 \cdot \frac{\text{Accuracy} \cdot \text{Source Accuracy}}{\text{Accuracy} + \text{Source Accuracy}}
$$

**Python Example:**
```python
def h_score(acc_t, acc_s):
    return 2 * acc_t * acc_s / (acc_t + acc_s + 1e-8)
```

---

## 7.2 Few-Shot Learning Metrics

### N-way K-shot Accuracy
Measures accuracy on query set given support set:
$$
\text{Accuracy} = \frac{1}{|\mathcal{Q}|} \sum_{(x,y) \in \mathcal{Q}} \mathbb{1}[\arg\max_c P(c|x, \mathcal{S}) = y]
$$

**Python Example:**
```python
def few_shot_accuracy(preds, labels):
    return (preds == labels).mean()
```

---

## 7.3 Zero-Shot Learning Metrics

### Top-1 Accuracy
Measures if the top prediction matches the true label:
$$
\text{Top-1 Accuracy} = \frac{1}{|\mathcal{D}_u|} \sum_{(x,y) \in \mathcal{D}_u} \mathbb{1}[\arg\max_c P(c|x) = y]
$$

### Harmonic Mean
Balances seen and unseen class accuracy:
$$
\text{Harmonic Mean} = \frac{2 \cdot \text{Seen Accuracy} \cdot \text{Unseen Accuracy}}{\text{Seen Accuracy} + \text{Unseen Accuracy}}
$$

**Python Example:**
```python
def harmonic_mean(acc_seen, acc_unseen):
    return 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen + 1e-8)
```

---

## Summary
- **Target Accuracy**: Main metric for domain adaptation
- **H-score**: Combines source and target accuracy
- **Few-Shot Accuracy**: Evaluates query set performance
- **Top-1 Accuracy**: Used in zero-shot learning
- **Harmonic Mean**: Balances seen/unseen class performance

These metrics help compare and benchmark models in transfer learning scenarios. 