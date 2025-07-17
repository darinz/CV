# Deep Learning for Computer Vision

> **Note: This repository is currently under construction. Content and structure may change as the curriculum is developed.**

---

Deep learning has fundamentally transformed computer vision, enabling machines to perceive and understand visual information with unprecedented accuracy. Computer vision has become ubiquitous in our society, with applications spanning search engines, image understanding systems, mobile applications, mapping services, medical imaging, drone technology, and autonomous vehicles. From recognizing objects in images to understanding complex scenes in videos, deep learning models have achieved human-level performance on many visual tasks.

Core to many of these applications are visual recognition tasks such as image classification, localization, and object detection. Recent developments in neural network approaches (commonly known as "deep learning") have dramatically advanced the performance of state-of-the-art visual recognition systems. This revolution has been driven by advances in convolutional neural networks, attention mechanisms, and generative models, opening new possibilities in autonomous systems, healthcare, entertainment, and beyond.

## Overview

This repository provides a comprehensive curriculum for deep learning in computer vision, covering fundamental concepts to advanced applications. The curriculum is structured to take you from basic linear classifiers through state-of-the-art architectures like Vision Transformers and diffusion models.

This curriculum is a deep dive into the details of deep learning architectures with a focus on learning end-to-end models for visual recognition tasks, particularly image classification. The learning journey begins with foundational concepts including linear classifiers, neural networks, and backpropagation. You'll then explore convolutional neural networks and their evolution through architectures like AlexNet, VGG, and ResNet. The curriculum advances to modern techniques including attention mechanisms, transformers, and self-supervised learning.

Practical applications are emphasized throughout, with hands-on experience in image classification, object detection, image segmentation, and video understanding. You'll also explore cutting-edge areas like generative models (VAEs, GANs, diffusion models), 3D vision, vision-language models, and robotic learning.

The curriculum balances theoretical understanding with practical implementation, ensuring you can both understand the underlying principles and build working systems. Through structured modules and real-world projects, you'll develop the skills needed to tackle current and future challenges in computer vision.

## Learning Objectives

- **Foundational Understanding**: Master the mathematical foundations of neural networks, backpropagation, and optimization techniques
- **CNN Mastery**: Build and train convolutional neural networks, understanding architectures from AlexNet to modern ResNet variants
- **Modern Architectures**: Implement and understand attention mechanisms, transformers, and self-supervised learning approaches
- **Computer Vision Tasks**: Develop expertise in image classification, object detection, semantic/instance segmentation, and video understanding
- **Generative AI**: Build and train generative models including VAEs, GANs, and diffusion models for image synthesis and manipulation
- **Advanced Applications**: Explore 3D vision, vision-language models, and robotic learning with deep reinforcement learning
- **Engineering Excellence**: Learn production-ready techniques for large-scale distributed training, model optimization, and deployment
- **Research Literacy**: Analyze and implement cutting-edge computer vision papers, staying current with the latest developments

## Deep Learning Basics

### [Image Classification with Linear Classifiers](01_Linear_Classifiers/)
- The data-driven approach
- K-nearest neighbor
- Linear Classifiers
- Algebraic / Visual / Geometric viewpoints
- Softmax loss

### [Regularization and Optimization](2gularization_Optimization/)
- Regularization
- Stochastic Gradient Descent
- Momentum, AdaGrad, Adam
- Learning rate schedules

### [Neural Networks and Backpropagation](3Neural_Networks/)
- Multi-layer Perceptron
- Backpropagation

## Perceiving and Understanding the Visual World

### [Image Classification with CNNs](4_CNNs/)
- History
- Higher-level representations, image features
- Convolution and pooling

### [CNN Architectures](5_CNN_Architectures/)
- Batch Normalization
- Transfer learning
- AlexNet, VGG, ResNet

### [Recurrent Neural Networks](6_RNN/)
- RNN, LSTM, GRU
- Language modeling
- Image captioning
- Sequence-to-sequence

### [Attention and Transformers](7nsformers/)
- Self-Attention
- Transformers

### [Multi-Modal Learning](08_Multi-Modal_Learning/)
- Vision-language models (CLIP, DALL-E, GPT-4V)
- Audio-visual learning
- Cross-modal retrieval and generation
- Multimodal fusion strategies
- Text-to-image and image-to-text models

### [Domain Adaptation and Transfer Learning](09_Domain_Adaptation/)
- Domain adaptation techniques
- Few-shot and zero-shot learning
- Meta-learning for computer vision
- Cross-domain generalization
- Domain-invariant representations

### [Object Detection, Image Segmentation, Visualizing and Understanding](10_Visualizing_Understanding/)
- Single-stage detectors
- Two-stage detectors
- Semantic/Instance/Panoptic segmentation
- Feature visualization and inversion
- Adversarial examples
- DeepDream and style transfer

### [Video Understanding](11_Video_Understanding/)
- Video classification
- 3D CNNs
- Two-stream networks
- Multimodal video understanding

### [Real-time and Streaming Applications](12_Real-Time_Applications/)
- Real-time object detection and tracking
- Video streaming analysis
- Edge computing for computer vision
- Latency optimization techniques
- Mobile and embedded deployment

### [Large Scale Distributed Training](13_Distributed_Training/)

## Generative and Interactive Visual Intelligence

### [Self-Supervised Learning](14_Self-Supervised_Learning/)
- Pretext tasks
- Contrastive learning
- Multisensory supervision

### [3D Vision](15_3D_Vision/)
- 3D shape representations
- Shape reconstruction
- Neural implicit representations

### [Generative Models](16_Generative_Models/)
- Variational Autoencoders
- Generative Adversarial Network
- Autoregressive Models 
- Diffusion models 

### [Vision and Language](17_Vision_Language/)

### [Robot Learning](18_Robot_Learning/)
- Deep Reinforcement Learning
- Model Learning
- Robotic Manipulation

## Technical Stack

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **NumPy/SciPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Jupyter Notebooks**: Interactive development

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Basic knowledge of linear algebra and calculus

### Installation
```bash
# Clone the repository
git clone https://github.com/darinz/CV.git
cd CV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Contributing

This is an educational repository. Contributions are welcome in the form of:
- Bug fixes
- Documentation improvements
- Additional examples
- Performance optimizations