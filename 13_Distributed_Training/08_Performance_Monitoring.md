# Performance Monitoring in Distributed Training

Monitoring performance is essential for diagnosing bottlenecks and optimizing distributed training.

## Training Metrics

### Throughput
- **Samples per Second:**
$$
\text{Throughput} = \frac{\text{total\_samples}}{\text{training\_time}}
$$
- **Tokens per Second:**
$$
\text{Token Throughput} = \frac{\text{total\_tokens}}{\text{training\_time}}
$$

**Python Example:**
```python
import time
start = time.time()
# Training loop...
end = time.time()
throughput = total_samples / (end - start)
```

### GPU Utilization
- **Utilization:**
$$
\text{Utilization} = \frac{\text{actual\_compute\_time}}{\text{total\_time}}
$$

**Python Example (nvidia-smi):**
```bash
watch -n 1 nvidia-smi
```

### Communication Efficiency
- **Comm Efficiency:**
$$
\text{Comm Efficiency} = \frac{T_{comp}}{T_{comp} + T_{comm}}
$$

## Profiling Tools

- **PyTorch Profiler:**
```python
import torch.profiler
with torch.profiler.profile() as prof:
    # Training code
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

- **Bandwidth Usage:**
$$
\text{Bandwidth Usage} = \frac{\text{data\_transferred}}{\text{time} \times \text{theoretical\_bandwidth}}
$$

---

Performance monitoring helps identify inefficiencies and guides optimization in distributed training. 