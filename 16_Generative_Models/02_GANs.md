# Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) are a class of generative models that learn to synthesize new data by pitting two neural networks against each other: a generator and a discriminator. GANs have revolutionized the field of generative modeling, especially for images.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Model Architecture](#model-architecture)
4. [Training Objective](#training-objective)
5. [Training Dynamics](#training-dynamics)
6. [Implementation](#implementation)
7. [Applications](#applications)
8. [Advantages and Limitations](#advantages-and-limitations)
9. [Advanced Topics](#advanced-topics)

## Introduction

GANs consist of two networks:
- **Generator (G):** Learns to map random noise to data space, generating fake samples.
- **Discriminator (D):** Learns to distinguish between real and fake samples.

The two networks are trained in a minimax game: the generator tries to fool the discriminator, while the discriminator tries to correctly classify real and fake samples.

## Mathematical Foundation

### Objective Function
The original GAN objective is:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p(z)} [\log(1 - D(G(z)))]
$$

- $x$ is a real data sample.
- $z$ is a random noise vector sampled from a prior (e.g., Gaussian).
- $G(z)$ is the generated (fake) sample.
- $D(x)$ outputs the probability that $x$ is real.

### Intuition
- The generator improves by making $D(G(z))$ close to 1 (fooling the discriminator).
- The discriminator improves by making $D(x)$ close to 1 for real data and $D(G(z))$ close to 0 for fake data.

## Model Architecture

### Generator
The generator is typically a neural network that takes a noise vector $z$ and outputs a data sample (e.g., an image):

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_fake = torch.tanh(self.fc3(h))  # For images scaled to [-1, 1]
        return x_fake
```

### Discriminator
The discriminator is a neural network that takes a data sample and outputs a probability (real or fake):

```python
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        out = torch.sigmoid(self.fc3(h))
        return out
```

## Training Objective

The loss functions for the discriminator and generator are:

```python
def discriminator_loss(real_scores, fake_scores):
    real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
    fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
    return real_loss + fake_loss

def generator_loss(fake_scores):
    return F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))
```

## Training Dynamics

- **Alternating Updates:** In each iteration, update $D$ to distinguish real/fake, then update $G$ to fool $D$.
- **Non-convergence:** GANs can be unstable; careful tuning and techniques like feature matching, label smoothing, and batch normalization help.

## Implementation

Here's a simple GAN implementation for MNIST:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
noise_dim = 100
hidden_dim = 256
image_dim = 784  # 28x28
batch_size = 128
lr = 0.0002
num_epochs = 20

# Models
generator = Generator(noise_dim, hidden_dim, image_dim)
discriminator = Discriminator(image_dim, hidden_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.view(-1, image_dim).to(device)
        batch_size = real_images.size(0)
        
        # Train Discriminator
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(z)
        real_scores = discriminator(real_images)
        fake_scores = discriminator(fake_images.detach())
        d_loss = discriminator_loss(real_scores, fake_scores)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(z)
        fake_scores = discriminator(fake_images)
        g_loss = generator_loss(fake_scores)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Generate and plot samples
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, noise_dim).to(device)
        samples = generator(z).view(-1, 28, 28).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    generator.train()
```

## Applications
- Image synthesis (e.g., faces, art)
- Data augmentation
- Super-resolution
- Image-to-image translation (e.g., CycleGAN, pix2pix)
- Text-to-image (e.g., DALL-E)

## Advantages and Limitations

### Advantages
- **High-Quality Samples:** GANs can generate sharp, realistic images.
- **Flexible Architectures:** Many variants for different tasks.
- **No Explicit Likelihood:** No need to specify a likelihood function.

### Limitations
- **Training Instability:** GANs are notoriously hard to train.
- **Mode Collapse:** Generator may produce limited variety.
- **No Explicit Density:** Cannot evaluate likelihood of samples.

## Advanced Topics

### 1. DCGAN (Deep Convolutional GAN)
Uses convolutional layers for image generation.

### 2. WGAN (Wasserstein GAN)
Uses Wasserstein distance for more stable training.

### 3. Conditional GAN (cGAN)
Generates data conditioned on labels.

### 4. StyleGAN
Generates high-resolution, high-quality images with style control.

### 5. CycleGAN
Unpaired image-to-image translation.

## Summary

GANs are a powerful class of generative models that have enabled major advances in image synthesis and beyond. While they can be challenging to train, their ability to generate high-quality, realistic samples has made them a cornerstone of modern generative modeling. 