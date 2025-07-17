# Evaluation Metrics for Multi-Modal Learning

Evaluation metrics are essential for measuring the performance of multi-modal models. This guide covers FID, IS, BLEU, ROUGE, R@K, and mAP, with math and code examples.

## 1. Introduction

Metrics help compare models and guide improvements. Different tasks require different metrics.

## 2. Image Generation Metrics

### 2.1 FID (Fréchet Inception Distance)

Measures similarity between generated and real images using feature statistics.

```math
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
```

#### Python Example: FID (using numpy)

```python
import numpy as np

def fid(mu_r, sigma_r, mu_g, sigma_g):
    diff = mu_r - mu_g
    covmean = np.linalg.sqrtm(sigma_r @ sigma_g)
    return diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
```

### 2.2 IS (Inception Score)

Measures both quality and diversity of generated images.

```math
\text{IS} = \exp(\mathbb{E}_{x} \text{KL}(p(y|x) \| p(y)))
```

## 3. Text Generation Metrics

### 3.1 BLEU Score

Measures n-gram overlap between generated and reference text.

```math
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
```

### 3.2 ROUGE Score

Measures recall of n-grams in generated text compared to references.

```math
\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}_{match}(gram_n)}{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}(gram_n)}
```

## 4. Cross-Modal Retrieval Metrics

### 4.1 R@K (Recall at K)

```math
\text{R@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{rank}_i \leq K]
```

### 4.2 mAP (Mean Average Precision)

```math
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i
```

#### Python Example: Recall@K

```python
def recall_at_k(ranks, k):
    return np.mean([rank <= k for rank in ranks])
```

## 5. Summary

Evaluation metrics like FID, IS, BLEU, ROUGE, R@K, and mAP are crucial for assessing the quality of multi-modal models in generation and retrieval tasks. 