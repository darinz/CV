# Generative Models

Generative models learn to model the underlying distribution of data, enabling the synthesis of new, realistic samples. They are foundational in unsupervised learning, image and text generation, and representation learning.

## Variational Autoencoders (VAEs)

VAEs are probabilistic generative models that learn a latent variable model for data $`x`$ with latent variables $`z`$.

### Model Structure
- **Encoder:** $`q_\phi(z|x)`$ approximates the posterior.
- **Decoder:** $`p_\theta(x|z)`$ generates data from latent code.

### Evidence Lower Bound (ELBO)
The VAE maximizes the ELBO:
```math
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
```
where $`D_{KL}`$ is the Kullback-Leibler divergence.

### Reparameterization Trick
Sample $`z`$ as:
```math
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```

## Generative Adversarial Networks (GANs)

GANs consist of a generator $`G`$ and a discriminator $`D`$ trained in a minimax game.

### Objective
```math
\min_G \max_D \; \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p(z)} [\log(1 - D(G(z)))]
```

- **Generator $`G(z)`$:** Maps noise $`z`$ to data space.
- **Discriminator $`D(x)`$:** Distinguishes real from generated samples.

### Training Dynamics
- $`G`$ tries to fool $`D`$; $`D`$ tries to distinguish real from fake.
- Common variants: DCGAN, WGAN, StyleGAN, CycleGAN.

## Autoregressive Models

Autoregressive models generate data one element at a time, modeling the joint distribution as a product of conditionals.

### Factorization
For data $`x = (x_1, x_2, ..., x_T)`$:
```math
p(x) = \prod_{t=1}^T p(x_t | x_{<t})
```

### Examples
- **PixelRNN/PixelCNN:** For images, model pixels sequentially.
- **WaveNet:** For audio, model waveform samples.
- **Transformer-based models:** For text and images (e.g., GPT, ImageGPT).

## Diffusion Models

Diffusion models generate data by reversing a gradual noising process.

### Forward Process
Add noise to data over $`T`$ steps:
```math
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
```

### Reverse Process
Learn to denoise step-by-step:
```math
p_\theta(x_{t-1} | x_t)
```

### Training Objective
Minimize the variational bound or a simplified denoising score matching loss:
```math
L = \mathbb{E}_{t, x_0, \epsilon} [\| \epsilon - \epsilon_\theta(x_t, t) \|^2]
```
where $`\epsilon`$ is the noise and $`\epsilon_\theta`$ is the predicted noise.

### Examples
- **DDPM (Denoising Diffusion Probabilistic Models)**
- **Score-based Generative Models**
- **Stable Diffusion**

## Applications
- Image, audio, and text synthesis
- Data augmentation
- Representation learning
- Inpainting, super-resolution, and style transfer

## Summary

Generative models provide powerful frameworks for learning data distributions and synthesizing new samples. VAEs, GANs, autoregressive, and diffusion models each offer unique strengths and have driven major advances in generative AI research and applications. 