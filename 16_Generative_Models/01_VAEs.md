# Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs) are probabilistic generative models that learn to encode data into a latent space and decode from it. They are particularly powerful for learning continuous latent representations and generating new data samples.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Model Architecture](#model-architecture)
4. [Training Objective](#training-objective)
5. [Reparameterization Trick](#reparameterization-trick)
6[Implementation](#implementation)
7. [Applications](#applications)
8es and Limitations](#advantages-and-limitations)

## Introduction

VAEs are based on the idea of variational inference, where we approximate a complex posterior distribution with a simpler one. The key insight is to learn a mapping from data to a latent space and back, while ensuring the latent space follows a known prior distribution (typically Gaussian).

### Key Components
- **Encoder (Recognition Network):** Maps input data to latent space parameters
- **Decoder (Generative Network):** Maps latent space back to data space
- **Latent Space:** Continuous representation where similar data points are close together

## Mathematical Foundation

### Problem Setup
Given data $x \sim p_{data}(x)$, we want to learn:
- A latent variable model $p_\theta(x, z) = p_\theta(x|z)p(z)$
- An approximate posterior $q_\phi(z|x)$

### Variational Inference
The true posterior $p(z|x)$ is intractable, so we approximate it with $q_\phi(z|x)$. We want to minimize the KL divergence:

$$D_[object Object]KL}(q_\phi(z|x) \| p(z|x))$$

However, this is also intractable. Instead, we maximize the Evidence Lower BOund (ELBO):

$$\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_[object Object]KL}(q_\phi(z|x) \| p(z))$$

### ELBO Components
1. **Reconstruction Term:** $\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)]$
   - Encourages the decoder to reconstruct the input well
2. **Regularization Term:** $-D_[object Object]KL}(q_\phi(z|x) \| p(z))$
   - Encourages the encoder to produce distributions close to the prior

## Model Architecture

### Encoder Network
The encoder $q_\phi(z|x)$ typically outputs:
- Mean: $\mu_\phi(x) \in \mathbb{R}^d$
- Log variance: $\log \sigma_\phi^2x) \in \mathbb{R}^d$

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

### Decoder Network
The decoder $p_\theta(x|z)$ reconstructs the input from latent code:

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2))
        x_recon = torch.sigmoid(self.fc3(h))  # For binary data
        return x_recon
```

## Training Objective

The VAE loss function combines reconstruction and regularization:

```python
def vae_loss(x_recon, x, mu, logvar):
   VAE loss function
    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    """
    # Reconstruction loss (binary cross entropy for binary data)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = 00.5 torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

## Reparameterization Trick

The reparameterization trick allows us to backpropagate through random sampling:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

```python
def reparameterize(mu, logvar):Reparameterization trick to sample from N(mu, var) from N(0,1   std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

## Implementation

Here's a complete VAE implementation for MNIST:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim=784en_dim=400, latent_dim=20:
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def sample(self, num_samples, device):
 erate samples from the VAE"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(z)

# Training function
def train_vae(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(-1784)  # Flatten MNIST images
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss = vae_loss(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(data):.6f}) 
    return train_loss / len(train_loader.dataset)

# Data loading and training
def main():
    # Hyperparameters
    batch_size =128   epochs =10
    lr =0.001    device = torch.device('cuda if torch.cuda.is_available() elsecpu)
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    for epoch in range(1, epochs +1        train_loss = train_vae(model, train_loader, optimizer, device, epoch)
        print(f'Epoch {epoch}: Average loss: {train_loss:.4f}')
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        samples = model.sample(16e)
        samples = samples.view(168)
        
        # Plot samples
        fig, axes = plt.subplots(4, 4, figsize=(8 8
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i], cmap='gray)           ax.axis(off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
```

## Applications

### 1. Image Generation
VAEs can generate new images by sampling from the learned latent space:

```python
def generate_images(model, num_samples=16device='cpu'):
 enerate new images from the VAE
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decoder(z)
        return samples.view(num_samples, 288```

### 2. Latent Space Interpolation
Interpolate between two points in latent space:

```python
def interpolate_latent_space(model, z1, z2num_steps=10):
nterpolate between two latent vectors"""
    alphas = torch.linspace(0,1teps)
    interpolated = []
    
    for alpha in alphas:
        z_interp = alpha * z1 + (1 - alpha) * z2
        with torch.no_grad():
            x_interp = model.decoder(z_interp)
            interpolated.append(x_interp)
    
    return torch.stack(interpolated)
```

### 3. Anomaly Detection
Use reconstruction error to detect anomalies:

```python
def detect_anomalies(model, data, threshold=00.1ct anomalies using reconstruction error
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(data)
        mse = F.mse_loss(recon, data, reduction=none').mean(dim=1)
        anomalies = mse > threshold
    return anomalies
```

## Advantages and Limitations

### Advantages
- **Continuous Latent Space:** Enables smooth interpolation and generation
- **Probabilistic Framework:** Provides uncertainty estimates
- **Regularization:** Built-in regularization through KL divergence
- **Interpretable:** Latent dimensions can capture meaningful features

### Limitations
- **Blurry Reconstructions:** Often produces blurry images compared to GANs
- **Posterior Collapse:** Encoder may ignore some latent dimensions
- **Training Instability:** Can be sensitive to hyperparameters
- **Limited Expressiveness:** May struggle with complex data distributions

## Advanced Topics

### β-VAE
A variant that adds a weight to the KL divergence term:

$$\mathcal{L}_\beta(x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - \beta \cdot D_[object Object]KL}(q_\phi(z|x) \| p(z))$$

```python
def beta_vae_loss(x_recon, x, mu, logvar, beta=1.0:
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction=sum)
    kl_loss = 00.5 torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

### Disentangled VAEs
Techniques to encourage disentangled representations:
- β-VAE
- FactorVAE
- β-TCVAE

### Conditional VAEs
VAEs that can generate data conditioned on additional information:

```python
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        self.encoder = ConditionalEncoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, condition_dim, hidden_dim, input_dim)
```

## Summary

VAEs provide a powerful framework for learning continuous latent representations and generating new data. They balance reconstruction quality with regularization through the ELBO objective. While they may produce blurrier outputs than GANs, they offer better theoretical foundations and more stable training. The reparameterization trick enables end-to-end training, and the probabilistic nature allows for uncertainty quantification and various applications in generative modeling. 