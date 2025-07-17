# Scaling Laws in Distributed Training

Scaling laws describe how model performance, training time, and efficiency change as you increase model size, data, or hardware resources.

## Model Scaling

### Parameter Scaling
- Increasing the number of parameters can improve model performance, but with diminishing returns.

**Chinchilla Scaling Law:**
$$
N_{opt} = 20 \times D^{0.7}
$$
where $N$ is the number of parameters and $D$ is the number of training tokens.

**Compute Requirement:**
$$
\text{FLOPs} = 6 \times N \times D
$$

## Hardware Scaling

### Multi-GPU Scaling
- **Strong Scaling:** Fixed total data, more GPUs = faster training.
- **Weak Scaling:** Increase data and GPUs proportionally, keep time per epoch constant.

**Amdahl's Law:**
$$
\text{Speedup} = \frac{1}{(1-p) + \frac{p}{N}}
$$
where $p$ is the parallelizable fraction.

**Python Example:**
```python
def amdahl_speedup(p, N):
    return 1 / ((1 - p) + p / N)

print(amdahl_speedup(0.95, 8))  # Example: 8 GPUs, 95% parallel
```

### Multi-Node Scaling
- Network bandwidth and latency become critical.

**Math:**
$$
\text{Required Bandwidth} = \frac{\text{model\_size} \times \text{updates\_per\_second}}{8}
$$

---

Understanding scaling laws helps design efficient distributed training systems and predict the impact of scaling up models or hardware. 