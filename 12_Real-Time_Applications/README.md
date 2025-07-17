# Real-time and Streaming Applications

This module explores techniques for deploying computer vision systems in real-time and streaming environments, focusing on latency optimization, edge computing, and mobile deployment.

## Real-time Object Detection and Tracking

Real-time object detection and tracking systems process video streams with minimal latency while maintaining accuracy.

### Real-time Detection Pipeline

#### Frame Processing Time

```math
T_{total} = T_{preprocess} + T_{inference} + T_{postprocess} + T_{tracking}
```

**Target FPS:**
```math
\text{FPS} = \frac{1}{T_{total}}
```

#### Latency Budget

```math
T_{budget} = \frac{1000}{\text{target\_fps}} \text{ ms}
```

### Real-time Object Detection

#### YOLO Real-time Variants

**YOLOv4-tiny:**
```math
\text{Input: } 416 \times 416 \times 3
```

```math
\text{Output: } \{(x, y, w, h, c, s)\}_{i=1}^{N}
```

**Inference Time:**
```math
T_{inference} = O(H \times W \times C \times K)
```

where $K$ is the number of convolutional operations.

#### MobileNet-SSD

**Depthwise Separable Convolution:**
```math
\text{Depthwise: } y_{i,j,k} = \sum_{m,n} x_{i+m,j+n,k} \cdot w_{m,n,k}
```

```math
\text{Pointwise: } z_{i,j,l} = \sum_{k} y_{i,j,k} \cdot v_{k,l}
```

**Computational Reduction:**
```math
\text{Reduction} = \frac{1}{K} + \frac{1}{C}
```

### Object Tracking

#### Kalman Filter

**State Prediction:**
```math
\hat{x}_t = F_t x_{t-1} + B_t u_t
```

```math
P_t = F_t P_{t-1} F_t^T + Q_t
```

**Measurement Update:**
```math
K_t = P_t H_t^T (H_t P_t H_t^T + R_t)^{-1}
```

```math
x_t = \hat{x}_t + K_t (z_t - H_t \hat{x}_t)
```

```math
P_t = (I - K_t H_t) P_t
```

#### SORT (Simple Online and Realtime Tracking)

**Hungarian Algorithm:**
```math
\text{Cost Matrix: } C_{ij} = 1 - \text{IoU}(b_i, b_j)
```

**Assignment:**
```math
\sigma^* = \arg\min_{\sigma} \sum_{i} C_{i,\sigma(i)}
```

#### DeepSORT

**Appearance Features:**
```math
f_i = \text{ReID}(b_i) \in \mathbb{R}^{128}
```

**Similarity:**
```math
\text{sim}(i, j) = \lambda \text{IoU}(b_i, b_j) + (1-\lambda) \cos(f_i, f_j)
```

### Multi-Object Tracking

#### Track Management

**Track State:**
```math
S_t \in \{\text{Tentative}, \text{Confirmed}, \text{Deleted}\}
```

**Track Score:**
```math
\text{Score}_t = \alpha \text{Score}_{t-1} + (1-\alpha) \text{Detection\_Confidence}
```

**Track Termination:**
```math
\text{Delete if } \text{Score}_t < \tau_{low} \text{ for } N_{miss} \text{ frames}
```

## Video Streaming Analysis

Video streaming analysis processes live video feeds with real-time constraints.

### Streaming Pipeline

#### Frame Buffer Management

**Buffer Size:**
```math
B_{size} = \text{max\_latency} \times \text{fps}
```

**Frame Drop Strategy:**
```math
\text{Drop if } \frac{B_{current}}{B_{size}} > \tau_{drop}
```

#### Adaptive Processing

**Quality Scaling:**
```math
Q_{scale} = \min(1.0, \frac{T_{budget}}{T_{current}})
```

**Resolution Scaling:**
```math
H_{new} = H_{original} \times \sqrt{Q_{scale}}
```

```math
W_{new} = W_{original} \times \sqrt{Q_{scale}}
```

### Real-time Video Analytics

#### Motion Detection

**Frame Difference:**
```math
D_t = \|I_t - I_{t-1}\|_2
```

**Motion Score:**
```math
M_t = \frac{1}{HW} \sum_{i,j} \mathbb{1}[D_t(i,j) > \tau_{motion}]
```

#### Background Subtraction

**Background Model:**
```math
B_t = \alpha B_{t-1} + (1-\alpha) I_t
```

**Foreground Detection:**
```math
F_t = \|I_t - B_t\| > \tau_{bg}
```

### Streaming Optimization

#### Temporal Sampling

**Adaptive Sampling:**
```math
\text{Skip frames if } \frac{T_{processing}}{T_{budget}} > 1.0
```

**Key Frame Detection:**
```math
\text{Key frame if } \|I_t - I_{last\_key}\| > \tau_{key}
```

## Edge Computing for Computer Vision

Edge computing brings computation closer to data sources, reducing latency and bandwidth requirements.

### Edge Deployment Architecture

#### Model Partitioning

**Cloud-Edge Split:**
```math
f_{total} = f_{edge} \circ f_{cloud}
```

**Latency Model:**
```math
T_{total} = T_{edge} + T_{network} + T_{cloud}
```

where:
```math
T_{network} = \frac{\text{data\_size}}{\text{bandwidth}} + \text{propagation\_delay}
```

#### Edge-Cloud Optimization

**Optimal Split:**
```math
\text{Split}^* = \arg\min_{\text{split}} T_{total}(\text{split})
```

### Model Compression

#### Quantization

**Uniform Quantization:**
```math
Q(x) = \text{round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \times (2^b - 1)\right)
```

**Quantization Error:**
```math
\epsilon_{quant} = \frac{x_{max} - x_{min}}{2^b}
```

#### Pruning

**Magnitude-based Pruning:**
```math
\text{Keep if } |w_{ij}| > \tau_{prune}
```

**Sparsity:**
```math
\text{Sparsity} = \frac{\text{zero\_weights}}{\text{total\_weights}}
```

#### Knowledge Distillation

**Teacher-Student Training:**
```math
L_{KD} = \alpha L_{CE}(y, \hat{y}_s) + (1-\alpha) L_{KL}(T_s, T_t)
```

where $T_s, T_t$ are soft targets.

### Edge Hardware Optimization

#### GPU Acceleration

**CUDA Kernels:**
```math
\text{Speedup} = \frac{T_{CPU}}{T_{GPU}}
```

**Memory Bandwidth:**
```math
\text{Bandwidth} = \frac{\text{data\_size}}{\text{transfer\_time}}
```

#### Tensor Cores

**Mixed Precision:**
```math
\text{FP16: } 16 \text{ bits per parameter}
```

```math
\text{Memory Reduction: } \frac{1}{2} \text{ compared to FP32}
```

## Latency Optimization Techniques

Latency optimization techniques reduce inference time while maintaining accuracy.

### Model Architecture Optimization

#### Efficient Convolutions

**Grouped Convolution:**
```math
\text{FLOPs} = \frac{H \times W \times C \times K \times K}{G}
```

where $G$ is the number of groups.

**Depthwise Separable:**
```math
\text{FLOPs} = H \times W \times (C + K \times K)
```

#### Neural Architecture Search (NAS)

**Search Space:**
```math
\mathcal{A} = \{\text{operations}, \text{connections}, \text{hyperparameters}\}
```

**Latency Constraint:**
```math
\text{Latency}(\mathcal{A}) \leq T_{target}
```

### Inference Optimization

#### Model Pruning

**Structured Pruning:**
```math
\text{Remove entire channels if } \|\mathbf{w}_c\|_2 < \tau
```

**Unstructured Pruning:**
```math
\text{Remove individual weights if } |w_{ij}| < \tau
```

#### Quantization

**Dynamic Quantization:**
```math
\text{Scale} = \frac{\text{max\_abs}}{\text{quant\_max}}
```

**Static Quantization:**
```math
\text{Calibrate on representative dataset}
```

### Memory Optimization

#### Memory Pooling

**Reuse Memory:**
```math
\text{Peak Memory} = \max_{l} \text{Memory}(l)
```

**Gradient Checkpointing:**
```math
\text{Memory} = O(\sqrt{L}) \text{ instead of } O(L)
```

#### Batch Processing

**Optimal Batch Size:**
```math
\text{Batch}^* = \arg\max_{b} \frac{\text{throughput}(b)}{b}
```

## Mobile and Embedded Deployment

Mobile and embedded deployment focuses on efficient execution on resource-constrained devices.

### Mobile Neural Networks

#### MobileNet Architecture

**Depthwise Separable Convolution:**
```math
\text{Parameters} = C \times K \times K + C \times C_{out}
```

**Computational Efficiency:**
```math
\text{FLOPs} = H \times W \times C \times (K \times K + C_{out})
```

#### ShuffleNet

**Channel Shuffle:**
```math
\text{Shuffle}(x) = \text{Reshape}(\text{Transpose}(\text{Reshape}(x)))
```

**Group Convolution:**
```math
\text{Groups} = \frac{C}{G}
```

### Model Optimization for Mobile

#### Mobile-Specific Pruning

**Channel Pruning:**
```math
\text{Remove channels with } \|\mathbf{w}_c\|_1 < \tau
```

**Filter Pruning:**
```math
\text{Remove filters with } \|\mathbf{w}_f\|_2 < \tau
```

#### Mobile Quantization

**INT8 Quantization:**
```math
\text{Memory Reduction: } \frac{1}{4} \text{ compared to FP32}
```

**Mixed Precision:**
```math
\text{Keep sensitive layers in FP16}
```

### Embedded System Optimization

#### ARM NEON Optimization

**SIMD Operations:**
```math
\text{Vector Operations: } 4 \text{ operations per cycle}
```

**Memory Alignment:**
```math
\text{Aligned access for optimal performance}
```

#### DSP Optimization

**Fixed-point Arithmetic:**
```math
\text{Q-format: } Q_m.n \text{ where } m+n = \text{word\_length}
```

**Look-up Tables:**
```math
\text{Replace expensive operations with LUT}
```

### Deployment Strategies

#### Model Conversion

**TensorFlow Lite:**
```math
\text{Model Size} = \text{Original Size} \times \text{compression\_ratio}
```

**ONNX Runtime:**
```math
\text{Cross-platform compatibility}
```

#### Runtime Optimization

**Operator Fusion:**
```math
\text{Fuse consecutive operations to reduce memory access}
```

**Parallel Execution:**
```math
\text{Utilize multiple cores for inference}
```

## Performance Metrics

### Latency Metrics

#### End-to-End Latency

```math
T_{e2e} = T_{input} + T_{preprocess} + T_{inference} + T_{postprocess} + T_{output}
```

#### Throughput

```math
\text{Throughput} = \frac{\text{number\_of\_inferences}}{\text{time}}
```

#### Real-time Factor (RTF)

```math
\text{RTF} = \frac{T_{processing}}{T_{input}}
```

**Real-time if RTF < 1.0**

### Accuracy Metrics

#### mAP at Different Latencies

```math
\text{mAP}(T) = \text{mAP at latency } T
```

#### Speed-Accuracy Trade-off

```math
\text{Pareto Frontier: } \{(T_i, \text{mAP}_i)\}_{i=1}^{N}
```

### Resource Metrics

#### Memory Usage

```math
\text{Peak Memory} = \max_{t} \text{Memory}(t)
```

#### Power Consumption

```math
\text{Power} = \text{Voltage} \times \text{Current}
```

```math
\text{Energy} = \text{Power} \times \text{Time}
```

## Applications

### Autonomous Vehicles

**Real-time Detection:**
```math
T_{detection} < 100 \text{ ms for safety}
```

**Multi-object Tracking:**
```math
\text{Track } N \text{ objects simultaneously}
```

### Surveillance Systems

**Motion Detection:**
```math
P(\text{motion}|I_t) = \sigma(f_{motion}(I_t))
```

**Object Tracking:**
```math
\text{Track objects across multiple cameras}
```

### Mobile Applications

**AR/VR:**
```math
T_{rendering} < 16.67 \text{ ms for 60 FPS}
```

**Image Recognition:**
```math
\text{Recognize objects in real-time}
```

### IoT Devices

**Smart Cameras:**
```math
\text{Process locally, send only alerts}
```

**Edge Analytics:**
```math
\text{Reduce cloud dependency}
```

## Summary

Real-time and streaming applications require careful optimization of:

1. **Real-time Detection and Tracking**: Fast object detection with efficient tracking algorithms
2. **Video Streaming Analysis**: Adaptive processing for live video feeds
3. **Edge Computing**: Bringing computation closer to data sources
4. **Latency Optimization**: Reducing inference time through various techniques
5. **Mobile Deployment**: Efficient execution on resource-constrained devices

These techniques enable computer vision systems to operate in real-time environments with strict latency requirements and limited computational resources. 