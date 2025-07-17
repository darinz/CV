# Performance Metrics for Real-Time Computer Vision

This guide explains how to measure and interpret performance metrics for real-time computer vision systems, with math and Python code examples.

## 1. Latency Metrics

### End-to-End Latency
Measures the total time from input to output.

```math
T_{e2e} = T_{input} + T_{preprocess} + T_{inference} + T_{postprocess} + T_{output}
```

#### Python Example: Measure Latency
```python
import time

def measure_latency(pipeline, input_data):
    start = time.time()
    output = pipeline(input_data)
    end = time.time()
    return end - start
```

### Throughput
Number of inferences per unit time.

```math
\text{Throughput} = \frac{\text{number\_of\_inferences}}{\text{time}}
```

#### Python Example: Measure Throughput
```python
def measure_throughput(pipeline, input_data, n_runs=100):
    import time
    start = time.time()
    for _ in range(n_runs):
        pipeline(input_data)
    end = time.time()
    return n_runs / (end - start)
```

### Real-time Factor (RTF)
Ratio of processing time to input time. Real-time if RTF < 1.0.

```math
\text{RTF} = \frac{T_{processing}}{T_{input}}
```

## 2. Accuracy Metrics

### mAP at Different Latencies
Mean Average Precision (mAP) is a standard detection metric, measured at various latencies.

```math
\text{mAP}(T) = \text{mAP at latency } T
```

#### Python Example: Compute mAP (using pycocotools)
```python
# Requires pycocotools and COCO-format data
from pycocotools.cocoeval import COCOeval

def compute_map(coco_gt, coco_dt):
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP
```

### Speed-Accuracy Trade-off
Plotting speed (latency) vs. accuracy (mAP) shows the Pareto frontier.

```math
\text{Pareto Frontier: } \{(T_i, \text{mAP}_i)\}_{i=1}^{N}
```

#### Python Example: Plot Speed-Accuracy
```python
import matplotlib.pyplot as plt

def plot_speed_accuracy(latencies, maps):
    plt.plot(latencies, maps, 'o-')
    plt.xlabel('Latency (ms)')
    plt.ylabel('mAP')
    plt.title('Speed-Accuracy Trade-off')
    plt.show()
```

## 3. Resource Metrics

### Memory Usage
Peak memory used during inference.

```math
\text{Peak Memory} = \max_{t} \text{Memory}(t)
```

#### Python Example: Measure Memory (PyTorch)
```python
import torch

def measure_peak_memory(model, input_tensor):
    torch.cuda.reset_peak_memory_stats()
    _ = model(input_tensor.cuda())
    return torch.cuda.max_memory_allocated()
```

### Power Consumption

```math
\text{Power} = \text{Voltage} \times \text{Current}
\text{Energy} = \text{Power} \times \text{Time}
```

#### Python Example: Read Power (NVIDIA GPU)
```python
import subprocess

def get_gpu_power():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
    return float(result.decode().strip())  # Watts
```

## Summary
- Latency, throughput, and RTF measure speed.
- mAP and speed-accuracy trade-off measure detection quality.
- Memory and power metrics are critical for real-time and embedded systems. 