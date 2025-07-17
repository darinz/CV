# Evaluation Metrics for RNNs

Evaluating sequence models requires specialized metrics. Two of the most common are BLEU and ROUGE, especially for tasks like translation and summarization.

## BLEU Score

BLEU (Bilingual Evaluation Understudy) measures the overlap between predicted and reference n-grams in machine translation.

### Mathematical Formulation

```math
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
```

Where:
- $`\text{BP}`$ is the brevity penalty
- $`p_n`$ is the n-gram precision
- $`w_n`$ are weights (often $`1/N`$)

**Python Example:**
```python
def ngram_precision(candidate, reference, n):
    from collections import Counter
    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    overlap = sum((cand_ngrams & ref_ngrams).values())
    total = max(sum(cand_ngrams.values()), 1)
    return overlap / total

def bleu(candidate, reference, max_n=4):
    import math
    precisions = [ngram_precision(candidate, reference, n) for n in range(1, max_n+1)]
    if min(precisions) == 0:
        return 0.0
    log_prec = sum((1/max_n) * math.log(p) for p in precisions)
    bp = min(1.0, math.exp(1 - len(reference)/len(candidate)))
    return bp * math.exp(log_prec)

# Example usage:
cand = ['the', 'cat', 'is', 'on', 'the', 'mat']
ref = ['the', 'cat', 'sat', 'on', 'the', 'mat']
print('BLEU score:', bleu(cand, ref))
```

## ROUGE Score

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap of n-grams, word sequences, and word pairs between the candidate and reference summaries.

### ROUGE-N

```math
\text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}_{match}(gram_n)}{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}(gram_n)}
```

**Python Example:**
```python
def rouge_n(candidate, reference, n):
    from collections import Counter
    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    overlap = sum((cand_ngrams & ref_ngrams).values())
    total = max(sum(ref_ngrams.values()), 1)
    return overlap / total

# Example usage:
cand = ['the', 'cat', 'is', 'on', 'the', 'mat']
ref = ['the', 'cat', 'sat', 'on', 'the', 'mat']
print('ROUGE-2 score:', rouge_n(cand, ref, 2))
```

---

**Summary:**
- **BLEU**: Precision-based, measures n-gram overlap for translation.
- **ROUGE**: Recall-based, measures n-gram overlap for summarization.

These metrics help compare model outputs to human references in a quantitative way. 