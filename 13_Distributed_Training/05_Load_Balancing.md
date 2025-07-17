# Load Balancing in Distributed Training

Load balancing ensures that all devices or workers are utilized efficiently, preventing bottlenecks and maximizing throughput.

## Dynamic Load Balancing

### Work Distribution
- Assigns work dynamically based on worker speed.
- Reduces idle time for faster workers.

**Math:**
$$
\text{Imbalance} = \frac{\max_i T_i - \min_i T_i}{\text{avg}(T_i)}
$$

### Adaptive Batch Sizing
- Adjusts batch size per worker based on speed.

**Math:**
$$
 b_i = \frac{\text{total\_batch\_size} \times \text{speed}_i}{\sum_j \text{speed}_j}
$$

**Python Example:**
```python
def adaptive_batch_sizes(total_batch_size, speeds):
    total_speed = sum(speeds)
    return [total_batch_size * s / total_speed for s in speeds]

# Example usage:
speeds = [1.0, 0.8, 1.2]
batch_sizes = adaptive_batch_sizes(120, speeds)
print(batch_sizes)  # Output: [40.0, 32.0, 48.0]
```

## Fault Tolerance

### Checkpointing and Recovery
- Save model state regularly to recover from failures.

**Python Example (PyTorch):**
```python
# Saving
torch.save(model.state_dict(), 'checkpoint.pth')
# Loading
model.load_state_dict(torch.load('checkpoint.pth'))
```

### Straggler Mitigation
- Replicate slow tasks on backup workers (speculative execution).

---

Load balancing and fault tolerance are key for robust, efficient distributed training, especially at scale. 