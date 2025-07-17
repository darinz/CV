# Object Detection, Image Segmentation, Visualizing and Understanding

This module explores advanced computer vision tasks including object detection, image segmentation, and techniques for understanding and visualizing deep neural networks.

## Object Detection

Object detection involves locating and classifying objects within images, typically using bounding boxes.

### Single-Stage Detectors

Single-stage detectors directly predict object locations and classes without region proposal steps.

#### YOLO (You Only Look Once)

YOLO divides the image into a grid and predicts bounding boxes for each grid cell.

**Grid-based Prediction:**
```math
P(\text{object}) \in [0, 1]
```

```math
(x, y, w, h) \in \mathbb{R}^4
```

```math
C_1, C_2, \ldots, C_K \in [0, 1]^K
```

**Loss Function:**
```math
L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]
```

```math
+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]
```

```math
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2
```

```math
+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
```

```math
+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
```

#### SSD (Single Shot MultiBox Detector)

SSD uses multiple feature maps at different scales for detection.

**Multi-scale Feature Maps:**
```math
F_l \in \mathbb{R}^{H_l \times W_l \times C_l}
```

**Default Boxes:**
```math
d_k = (s_k, s_k, \sqrt{\frac{a_r}{a_r}}, \sqrt{a_r \cdot a_r})
```

**Prediction:**
```math
(l, g, s) = \text{SSD}(F_1, F_2, \ldots, F_L)
```

where $l$ is location offsets, $g$ is confidence scores, and $s$ is class scores.

#### RetinaNet

RetinaNet introduces Focal Loss to address class imbalance.

**Focal Loss:**
```math
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
```

where:
- $p_t = p$ if $y = 1$, else $p_t = 1 - p$
- $\alpha_t$ is the balancing parameter
- $\gamma$ is the focusing parameter

**Classification Loss:**
```math
L_{cls} = \frac{1}{N} \sum_{i} \text{FL}(p_i, y_i)
```

**Regression Loss:**
```math
L_{reg} = \frac{1}{N} \sum_{i} \mathbb{1}_{[y_i = 1]} \text{smooth}_{L1}(t_i - \hat{t}_i)
```

### Two-Stage Detectors

Two-stage detectors first propose regions of interest, then classify and refine them.

#### R-CNN (Region-based CNN)

**Region Proposal:**
```math
R = \text{SelectiveSearch}(I)
```

**Feature Extraction:**
```math
f_i = \text{CNN}(\text{crop}(I, r_i))
```

**Classification:**
```math
P(c|f_i) = \text{softmax}(W_c f_i + b_c)
```

#### Fast R-CNN

**RoI Pooling:**
```math
F_{roi} = \text{RoIPool}(F, r)
```

**Multi-task Loss:**
```math
L = L_{cls} + \lambda L_{reg}
```

where:
```math
L_{cls} = -\log P(c^*|F_{roi})
```

```math
L_{reg} = \text{smooth}_{L1}(t - t^*)
```

#### Faster R-CNN

**Region Proposal Network (RPN):**
```math
P(\text{object}) = \sigma(W_{cls} F + b_{cls})
```

```math
t = W_{reg} F + b_{reg}
```

**Anchor Boxes:**
```math
A = \{(s_i, r_j) | s_i \in S, r_j \in R\}
```

where $S$ are scales and $R$ are aspect ratios.

## Image Segmentation

Image segmentation involves partitioning images into meaningful regions.

### Semantic Segmentation

Semantic segmentation assigns a class label to each pixel.

#### FCN (Fully Convolutional Networks)

**Upsampling:**
```math
F_{up} = \text{upsample}(F, \text{scale})
```

**Skip Connections:**
```math
F_{skip} = F_{up} + F_{skip}
```

**Pixel-wise Classification:**
```math
P(c|p_{ij}) = \text{softmax}(W_c F_{ij} + b_c)
```

#### U-Net

**Encoder-Decoder Architecture:**
```math
F_{enc}^l = \text{Encoder}_l(F_{enc}^{l-1})
```

```math
F_{dec}^l = \text{Decoder}_l(F_{dec}^{l+1}, F_{enc}^l)
```

**Skip Connections:**
```math
F_{skip}^l = \text{Concat}(F_{dec}^l, F_{enc}^l)
```

#### DeepLab

**Atrous Convolution:**
```math
y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]
```

where $r$ is the dilation rate.

**ASPP (Atrous Spatial Pyramid Pooling):**
```math
F_{aspp} = \text{Concat}(\text{Conv}_{r=6}, \text{Conv}_{r=12}, \text{Conv}_{r=18}, \text{GlobalAvgPool})
```

### Instance Segmentation

Instance segmentation distinguishes between different instances of the same class.

#### Mask R-CNN

**RoI Align:**
```math
F_{roi} = \text{RoIAlign}(F, r)
```

**Mask Head:**
```math
M = \text{MaskHead}(F_{roi})
```

**Mask Loss:**
```math
L_{mask} = -\frac{1}{m^2} \sum_{i,j} [y_{ij} \log(\hat{y}_{ij}) + (1 - y_{ij}) \log(1 - \hat{y}_{ij})]
```

#### YOLACT

**Protonet:**
```math
P_k = \text{Protonet}(F)
```

**Prediction Head:**
```math
c_k = \text{PredictionHead}(F)
```

**Mask Assembly:**
```math
M = \sum_{k} c_k \cdot P_k
```

### Panoptic Segmentation

Panoptic segmentation combines semantic and instance segmentation.

#### Panoptic FPN

**Semantic Branch:**
```math
S = \text{SemanticHead}(F)
```

**Instance Branch:**
```math
I = \text{InstanceHead}(F)
```

**Panoptic Fusion:**
```math
P = \text{Fusion}(S, I)
```

## Feature Visualization and Inversion

Understanding what neural networks learn through visualization and inversion techniques.

### Feature Visualization

#### Activation Maximization

Find input that maximizes neuron activation:

```math
x^* = \arg\max_x f_l(x) - \lambda \|x\|^2
```

where $f_l(x)$ is the activation of layer $l$.

#### Gradient Ascent

```math
x_{t+1} = x_t + \alpha \nabla_x f_l(x_t)
```

#### Regularization

**L2 Regularization:**
```math
R(x) = \lambda \|x\|^2
```

**Total Variation:**
```math
R(x) = \lambda \sum_{i,j} \sqrt{(x_{i+1,j} - x_{i,j})^2 + (x_{i,j+1} - x_{i,j})^2}
```

**Frequency Penalty:**
```math
R(x) = \lambda \|\mathcal{F}(x)\|^2
```

### Feature Inversion

#### Inversion Problem

Find input that produces given feature representation:

```math
x^* = \arg\min_x \|f_l(x) - f_l(x_0)\|^2 + R(x)
```

#### Iterative Optimization

```math
x_{t+1} = x_t - \alpha \nabla_x [\|f_l(x_t) - f_l(x_0)\|^2 + R(x_t)]
```

#### Progressive Reconstruction

```math
x_{t+1} = \text{Proj}(x_t - \alpha \nabla_x L(x_t))
```

where $\text{Proj}$ projects to valid image space.

### Saliency Maps

#### Gradient-based Saliency

```math
S_{ij} = \left|\frac{\partial f_c}{\partial x_{ij}}\right|
```

#### Guided Backpropagation

```math
\frac{\partial f_c}{\partial x_{ij}} = \begin{cases}
\frac{\partial f_c}{\partial x_{ij}} & \text{if } \frac{\partial f_c}{\partial x_{ij}} > 0 \\
0 & \text{otherwise}
\end{cases}
```

#### Grad-CAM

```math
\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial f_c}{\partial A_{ij}^k}
```

```math
L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
```

## Adversarial Examples

Adversarial examples are inputs designed to fool neural networks.

### Adversarial Attack Formulation

```math
x_{adv} = x + \delta
```

subject to:
```math
\|\delta\|_p \leq \epsilon
```

```math
f(x_{adv}) \neq f(x)
```

### Fast Gradient Sign Method (FGSM)

```math
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))
```

### Projected Gradient Descent (PGD)

```math
x_{t+1} = \text{Proj}_{B_\epsilon(x)} (x_t + \alpha \cdot \text{sign}(\nabla_x L(x_t, y)))
```

where $B_\epsilon(x) = \{x' : \|x' - x\|_\infty \leq \epsilon\}$.

### Carlini & Wagner (C&W) Attack

```math
\min_{\delta} \|\delta\|_2^2 + c \cdot f(x + \delta)
```

where $f$ is the objective function for misclassification.

### Adversarial Training

**Min-Max Optimization:**
```math
\min_\theta \max_{\|\delta\| \leq \epsilon} L(f_\theta(x + \delta), y)
```

**Adversarial Loss:**
```math
L_{adv} = L(f_\theta(x), y) + \lambda L(f_\theta(x + \delta), y)
```

## DeepDream and Style Transfer

### DeepDream

DeepDream amplifies patterns that the network recognizes.

#### Objective Function

```math
L(x) = \|f_l(x) - f_l(x_0)\|^2
```

#### Optimization

```math
x_{t+1} = x_t + \alpha \nabla_x L(x_t)
```

#### Octave Processing

```math
x_{octave} = \text{resize}(x, \text{scale})
```

```math
x_{result} = \text{resize}(x_{octave}, \text{original size})
```

### Neural Style Transfer

Neural style transfer combines content and style from different images.

#### Content Loss

```math
L_{content} = \frac{1}{2} \sum_{i,j} (F_{ij}^l - P_{ij}^l)^2
```

where $F^l$ and $P^l$ are feature maps of generated and content images.

#### Style Loss

**Gram Matrix:**
```math
G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l
```

**Style Loss:**
```math
L_{style} = \sum_l w_l \sum_{i,j} (G_{ij}^l - A_{ij}^l)^2
```

where $A^l$ is the Gram matrix of the style image.

#### Total Loss

```math
L_{total} = \alpha L_{content} + \beta L_{style}
```

#### Optimization

```math
x_{t+1} = x_t - \alpha \nabla_x L_{total}(x_t)
```

### Fast Neural Style Transfer

#### Style Network

```math
f_s: \mathcal{X} \rightarrow \mathcal{X}
```

**Training Objective:**
```math
L = L_{content} + \lambda L_{style}
```

#### Instance Normalization

```math
\text{IN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

where $\mu$ and $\sigma^2$ are computed across spatial dimensions.

## Evaluation Metrics

### Object Detection

#### mAP (Mean Average Precision)

```math
\text{AP} = \int_0^1 P(r) dr
```

```math
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
```

#### IoU (Intersection over Union)

```math
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
```

### Segmentation

#### Pixel Accuracy

```math
\text{Accuracy} = \frac{\sum_{i=1}^{n} p_{ii}}{\sum_{i=1}^{k} \sum_{j=1}^{k} p_{ij}}
```

#### Mean IoU

```math
\text{mIoU} = \frac{1}{k} \sum_{i=1}^{k} \frac{p_{ii}}{\sum_{j=1}^{k} p_{ij} + \sum_{j=1}^{k} p_{ji} - p_{ii}}
```

#### Dice Coefficient

```math
\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}
```

### Adversarial Robustness

#### Robust Accuracy

```math
\text{Robust Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[f(x_i^{adv}) = y_i]
```

#### Attack Success Rate

```math
\text{ASR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[f(x_i^{adv}) \neq f(x_i)]
```

## Applications

### Autonomous Driving

**Object Detection:**
```math
f: \mathcal{X} \rightarrow \{(b_i, c_i, s_i)\}_{i=1}^{N}
```

**Semantic Segmentation:**
```math
f: \mathcal{X} \rightarrow \mathcal{Y}^{H \times W}
```

### Medical Imaging

**Tumor Detection:**
```math
P(\text{tumor}|x) = \sigma(f_\theta(x))
```

**Organ Segmentation:**
```math
f: \mathcal{X} \rightarrow \{0, 1\}^{H \times W}
```

### Security

**Adversarial Defense:**
```math
f_{robust}: \mathcal{X} \rightarrow \mathcal{Y}
```

**Anomaly Detection:**
```math
P(\text{anomaly}|x) = 1 - P(\text{normal}|x)
```

## Summary

Object detection, segmentation, and visualization techniques are fundamental to computer vision:

1. **Object Detection**: Single-stage and two-stage detectors for locating objects
2. **Image Segmentation**: Semantic, instance, and panoptic segmentation
3. **Feature Visualization**: Understanding what networks learn
4. **Adversarial Examples**: Security vulnerabilities and defenses
5. **Style Transfer**: Artistic image generation and manipulation

These techniques enable advanced computer vision applications and provide insights into neural network behavior. 