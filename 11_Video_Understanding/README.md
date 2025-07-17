# Video Understanding

This module explores techniques for understanding and analyzing video content, including temporal dynamics, spatial-temporal features, and multimodal information processing.

## Video Classification

Video classification involves categorizing video content into predefined classes, considering both spatial and temporal information.

### Problem Formulation

Given a video sequence $V = \{v_1, v_2, \ldots, v_T\}$ where $v_t \in \mathbb{R}^{H \times W \times C}$:

```math
f: \mathcal{V} \rightarrow \mathcal{Y}
```

where $\mathcal{V}$ is the video space and $\mathcal{Y}$ is the class space.

### Temporal Modeling

#### Frame-level Features

Extract features from individual frames:

```math
h_t = f_\theta(v_t) \in \mathbb{R}^d
```

#### Temporal Aggregation

**Mean Pooling:**
```math
h_{video} = \frac{1}{T} \sum_{t=1}^{T} h_t
```

**Max Pooling:**
```math
h_{video} = \max_{t=1,\ldots,T} h_t
```

**Attention-based Aggregation:**
```math
\alpha_t = \frac{\exp(w^T h_t)}{\sum_{t'=1}^{T} \exp(w^T h_{t'})}
```

```math
h_{video} = \sum_{t=1}^{T} \alpha_t h_t
```

### Video-level Classification

```math
P(y|V) = \text{softmax}(W_{out} h_{video} + b_{out})
```

**Loss Function:**
```math
L = -\sum_{i=1}^{N} \log P(y_i|V_i)
```

## 3D CNNs

3D Convolutional Neural Networks extend 2D CNNs to process spatial-temporal data directly.

### 3D Convolution

#### Basic 3D Convolution

```math
y_{i,j,k} = \sum_{c=1}^{C} \sum_{t=0}^{T-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{c,i+t,j+h,k+w} \cdot w_{c,t,h,w}
```

where:
- $x \in \mathbb{R}^{C \times T \times H \times W}$ is the input
- $w \in \mathbb{R}^{C \times T \times H \times W}$ is the 3D kernel
- $y \in \mathbb{R}^{T' \times H' \times W'}$ is the output

#### 3D Convolution with Stride

```math
y_{i,j,k} = \sum_{c=1}^{C} \sum_{t=0}^{T-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{c,s_t \cdot i+t,s_h \cdot j+h,s_w \cdot k+w} \cdot w_{c,t,h,w}
```

where $s_t, s_h, s_w$ are temporal and spatial strides.

### C3D Architecture

#### Basic C3D Block

```math
\text{C3D}(x) = \text{ReLU}(\text{BatchNorm}(\text{Conv3D}(x)))
```

#### C3D Network Structure

**Input:** $V \in \mathbb{R}^{3 \times 16 \times 112 \times 112}$

**Layer 1:** Conv3D(64, 3×3×3) → ReLU → MaxPool3D(1×2×2)
```math
\text{Output: } 64 \times 16 \times 56 \times 56
```

**Layer 2:** Conv3D(128, 3×3×3) → ReLU → MaxPool3D(2×2×2)
```math
\text{Output: } 128 \times 8 \times 28 \times 28
```

**Layer 3:** Conv3D(256, 3×3×3) → ReLU → MaxPool3D(2×2×2)
```math
\text{Output: } 256 \times 4 \times 14 \times 14
```

**Layer 4:** Conv3D(512, 3×3×3) → ReLU → MaxPool3D(2×2×2)
```math
\text{Output: } 512 \times 2 \times 7 \times 7
```

**Layer 5:** Conv3D(512, 3×3×3) → ReLU → MaxPool3D(2×2×2)
```math
\text{Output: } 512 \times 1 \times 4 \times 4
```

### I3D (Inflated 3D ConvNet)

I3D inflates 2D filters to 3D by repeating them across the temporal dimension.

#### Filter Inflation

```math
W_{3D} = W_{2D} \otimes \mathbf{1}_T
```

where $\otimes$ denotes outer product and $\mathbf{1}_T$ is a vector of ones.

#### I3D Architecture

**Inception Module:**
```math
\text{Inception3D}(x) = \text{Concat}(\text{Branch1}(x), \text{Branch2}(x), \text{Branch3}(x), \text{Branch4}(x))
```

**Branch Operations:**
```math
\text{Branch1}(x) = \text{Conv3D}(64, 1×1×1)(x)
```

```math
\text{Branch2}(x) = \text{Conv3D}(96, 1×1×1) \circ \text{Conv3D}(128, 3×3×3)(x)
```

```math
\text{Branch3}(x) = \text{Conv3D}(16, 1×1×1) \circ \text{Conv3D}(32, 3×3×3) \circ \text{Conv3D}(32, 3×3×3)(x)
```

```math
\text{Branch4}(x) = \text{MaxPool3D}(3×3×3) \circ \text{Conv3D}(32, 1×1×1)(x)
```

### SlowFast Networks

SlowFast networks use two pathways with different temporal resolutions.

#### Slow Pathway

```math
x_S \in \mathbb{R}^{3 \times T \times H \times W}
```

**Temporal Sampling:**
```math
T_S = \frac{T_F}{\alpha}
```

where $\alpha$ is the temporal stride ratio.

#### Fast Pathway

```math
x_F \in \mathbb{R}^{3 \times T_F \times H \times W}
```

**Lateral Connections:**
```math
x_F' = \text{LateralConv}(x_F) + \text{Resample}(x_S)
```

#### Fusion

```math
h = \text{Fusion}(\text{SlowPath}(x_S), \text{FastPath}(x_F))
```

## Two-Stream Networks

Two-stream networks process spatial and temporal information separately and then fuse them.

### Spatial Stream

Process individual frames for appearance information:

```math
h_t^{spatial} = f_{spatial}(v_t) \in \mathbb{R}^d
```

**Temporal Aggregation:**
```math
h^{spatial} = \frac{1}{T} \sum_{t=1}^{T} h_t^{spatial}
```

### Temporal Stream

Process optical flow for motion information:

```math
f_t = \text{OpticalFlow}(v_t, v_{t+1}) \in \mathbb{R}^{H \times W \times 2}
```

```math
h_t^{temporal} = f_{temporal}(f_t) \in \mathbb{R}^d
```

**Temporal Aggregation:**
```math
h^{temporal} = \frac{1}{T-1} \sum_{t=1}^{T-1} h_t^{temporal}
```

### Fusion Strategies

#### Late Fusion

```math
h_{fused} = \text{Concat}(h^{spatial}, h^{temporal})
```

```math
P(y|V) = \text{softmax}(W_{fusion} h_{fused} + b_{fusion})
```

#### Weighted Fusion

```math
h_{fused} = \alpha h^{spatial} + (1-\alpha) h^{temporal}
```

where $\alpha$ is learned or fixed.

#### Attention Fusion

```math
\alpha = \sigma(W_a [h^{spatial}; h^{temporal}] + b_a)
```

```math
h_{fused} = \alpha h^{spatial} + (1-\alpha) h^{temporal}
```

### Optical Flow Computation

#### Lucas-Kanade Method

```math
\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}
```

where $I_x, I_y, I_t$ are spatial and temporal derivatives.

#### DeepFlow

```math
f = \text{DeepFlow}(v_t, v_{t+1}) = \text{CNN}([v_t, v_{t+1}])
```

## Multimodal Video Understanding

Multimodal video understanding combines multiple modalities (visual, audio, text) for comprehensive video analysis.

### Audio-Visual Fusion

#### Audio Features

**Mel-frequency Cepstral Coefficients (MFCC):**
```math
\text{MFCC} = \text{DCT}(\log(\text{MelFilterbank}(|\text{FFT}(x)|^2)))
```

**Audio Embedding:**
```math
h^{audio} = f_{audio}(a) \in \mathbb{R}^d
```

#### Cross-Modal Attention

```math
\alpha_{t,i} = \frac{\exp(\text{sim}(h_t^{visual}, h_i^{audio}))}{\sum_{j} \exp(\text{sim}(h_t^{visual}, h_j^{audio}))}
```

```math
h_t^{attended} = \sum_{i} \alpha_{t,i} h_i^{audio}
```

### Video-Language Understanding

#### Video Captioning

**Encoder-Decoder Architecture:**
```math
h^{video} = \text{VideoEncoder}(V)
```

```math
P(w_t|w_{<t}, V) = \text{Decoder}(h_{t-1}, h^{video})
```

#### Video Question Answering

**Question Encoding:**
```math
h^{question} = \text{QuestionEncoder}(Q)
```

**Video-Question Fusion:**
```math
h^{fused} = \text{CrossAttention}(h^{video}, h^{question})
```

**Answer Prediction:**
```math
P(a|V, Q) = \text{softmax}(W_{out} h^{fused} + b_{out})
```

### Multimodal Fusion Architectures

#### Early Fusion

```math
h_{early} = \text{Fusion}(h^{visual}, h^{audio}, h^{text})
```

```math
y = \text{Classifier}(h_{early})
```

#### Late Fusion

```math
y^{visual} = \text{Classifier}_{visual}(h^{visual})
```

```math
y^{audio} = \text{Classifier}_{audio}(h^{audio})
```

```math
y^{text} = \text{Classifier}_{text}(h^{text})
```

```math
y = \text{Ensemble}(y^{visual}, y^{audio}, y^{text})
```

#### Hierarchical Fusion

```math
h_{local} = \text{LocalFusion}(h^{visual}, h^{audio})
```

```math
h_{global} = \text{GlobalFusion}(h_{local}, h^{text})
```

```math
y = \text{Classifier}(h_{global})
```

### Temporal Modeling in Multimodal Context

#### LSTM-based Fusion

```math
h_t^{fused} = \text{LSTM}([h_t^{visual}, h_t^{audio}, h_t^{text}], h_{t-1}^{fused})
```

#### Transformer-based Fusion

```math
X = [h^{visual}, h^{audio}, h^{text}]
```

```math
h^{fused} = \text{Transformer}(X)
```

#### Cross-Modal Transformer

```math
h^{visual'} = \text{CrossAttention}(h^{visual}, h^{audio})
```

```math
h^{audio'} = \text{CrossAttention}(h^{audio}, h^{visual})
```

```math
h^{fused} = \text{Concat}(h^{visual'}, h^{audio'})
```

## Advanced Video Understanding

### Action Recognition

#### Temporal Action Localization

**Action Proposal:**
```math
P(\text{action}|t) = \sigma(f_{proposal}(h_t))
```

**Temporal Boundary:**
```math
(t_{start}, t_{end}) = \arg\max_{t_s, t_e} P(\text{action}|t_s, t_e)
```

#### Action Classification

```math
P(a|V) = \text{softmax}(W_{action} h^{video} + b_{action})
```

### Video Summarization

#### Key Frame Selection

```math
s_t = \sigma(f_{importance}(h_t))
```

```math
\text{Summary} = \{v_t | s_t > \tau\}
```

#### Video Summarization Loss

```math
L_{summary} = -\sum_{t} s_t \log P(y_t|v_t) + \lambda \sum_{t} s_t
```

### Video Retrieval

#### Video Embedding

```math
e^{video} = \text{Normalize}(f_{embed}(h^{video}))
```

#### Similarity Computation

```math
\text{sim}(V_1, V_2) = e_1^{video} \cdot e_2^{video}
```

## Training Strategies

### Temporal Sampling

#### Uniform Sampling

```math
\mathcal{T} = \{t_1, t_2, \ldots, t_K\} \text{ where } t_i = \frac{T}{K} \cdot i
```

#### Random Sampling

```math
\mathcal{T} = \{t_1, t_2, \ldots, t_K\} \text{ where } t_i \sim \text{Uniform}(1, T)
```

#### Segment-based Sampling

```math
\mathcal{T} = \bigcup_{i=1}^{N} \{t_{i,1}, t_{i,2}, \ldots, t_{i,K/N}\}
```

### Data Augmentation

#### Temporal Augmentation

**Temporal Cropping:**
```math
V' = \{v_t | t \in [t_{start}, t_{end}]\}
```

**Temporal Jittering:**
```math
t' = t + \delta \text{ where } \delta \sim \mathcal{N}(0, \sigma^2)
```

#### Spatial Augmentation

**Random Cropping:**
```math
v_t' = \text{crop}(v_t, \text{random box})
```

**Random Flipping:**
```math
v_t' = \text{flip}(v_t, \text{horizontal})
```

## Evaluation Metrics

### Video Classification

#### Top-1 Accuracy

```math
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max_c P(c|V_i) = y_i]
```

#### Top-5 Accuracy

```math
\text{Top-5 Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i \in \text{top5}(P(c|V_i))]
```

### Action Recognition

#### mAP (Mean Average Precision)

```math
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
```

#### Temporal IoU

```math
\text{tIoU} = \frac{|I_1 \cap I_2|}{|I_1 \cup I_2|}
```

where $I_1, I_2$ are temporal intervals.

### Video Retrieval

#### Recall@K

```math
\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{rank}_i \leq K]
```

#### Mean Reciprocal Rank (MRR)

```math
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
```

## Applications

### Video Surveillance

**Activity Recognition:**
```math
P(a|V) = \text{softmax}(W_{activity} h^{video} + b_{activity})
```

**Anomaly Detection:**
```math
P(\text{anomaly}|V) = \sigma(f_{anomaly}(h^{video}))
```

### Video Recommendation

**User Preference Modeling:**
```math
P(\text{like}|V, u) = \sigma(f_{preference}(h^{video}, h^{user}))
```

**Content Similarity:**
```math
\text{sim}(V_1, V_2) = \cos(e_1^{video}, e_2^{video})
```

### Video Editing

**Shot Boundary Detection:**
```math
P(\text{boundary}|t) = \sigma(f_{boundary}(h_t, h_{t+1}))
```

**Content-aware Editing:**
```math
V' = \text{Edit}(V, \text{style}, \text{content})
```

## Summary

Video understanding encompasses multiple techniques for analyzing temporal and multimodal content:

1. **Video Classification**: Categorizing video content using temporal modeling
2. **3D CNNs**: Processing spatial-temporal data with 3D convolutions
3. **Two-Stream Networks**: Combining spatial and temporal information
4. **Multimodal Video Understanding**: Integrating visual, audio, and text modalities

These techniques enable comprehensive video analysis for various applications including surveillance, recommendation, and content creation. 