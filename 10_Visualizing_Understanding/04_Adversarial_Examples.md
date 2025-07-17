# Adversarial Examples

Adversarial examples are inputs intentionally designed to fool neural networks, revealing vulnerabilities in deep learning models. This guide covers adversarial attack methods and defenses, with detailed explanations, math, and Python code examples.

## 1. Overview

- **Adversarial Attack**: Slightly perturbing input data to cause misclassification.
- **Adversarial Training**: Training models to be robust against such attacks.

---

## 2. Adversarial Attack Formulation

Given an input $x$ and label $y$, an adversarial example $x_{adv}$ is:

```math
x_{adv} = x + \delta
```
subject to:
```math
\|\delta\|_p \leq \epsilon
```
where $\epsilon$ is a small value, and $f(x_{adv}) \neq f(x)$.

---

## 3. Fast Gradient Sign Method (FGSM)

FGSM is a simple, efficient attack that perturbs the input in the direction of the gradient of the loss with respect to the input.

```math
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(x, y))
```

**Python Example (FGSM):**
```python
import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(pretrained=True).eval()
img = Image.open('image.jpg')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
x = preprocess(img).unsqueeze(0)
x.requires_grad = True

y = torch.tensor([target_class])  # Replace with true label
output = model(x)
loss = torch.nn.functional.cross_entropy(output, y)
loss.backward()

epsilon = 0.03
x_adv = x + epsilon * x.grad.sign()
```

---

## 4. Projected Gradient Descent (PGD)

PGD is an iterative version of FGSM, projecting the perturbed input back into the allowed $\epsilon$-ball after each step.

```math
x_{t+1} = \text{Proj}_{B_\epsilon(x)} (x_t + \alpha \cdot \text{sign}(\nabla_x L(x_t, y)))
```

**Python Example (PGD):**
```python
def pgd_attack(model, x, y, epsilon=0.03, alpha=0.01, iters=40):
    x_adv = x.clone().detach().requires_grad_(True)
    for i in range(iters):
        output = model(x_adv)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True)
    return x_adv
```

---

## 5. Carlini & Wagner (C&W) Attack

C&W attack formulates the attack as an optimization problem:

```math
\min_{\delta} \|\delta\|_2^2 + c \cdot f(x + \delta)
```
where $f$ is a function that encourages misclassification.

**Python Example:**
- The C&W attack is complex and typically implemented in libraries like [Foolbox](https://github.com/bethgelab/foolbox) or [CleverHans](https://github.com/cleverhans-lab/cleverhans).

---

## 6. Adversarial Training

Adversarial training improves robustness by including adversarial examples during training.

**Min-Max Optimization:**
```math
\min_\theta \max_{\|\delta\| \leq \epsilon} L(f_\theta(x + \delta), y)
```

**Adversarial Loss:**
```math
L_{adv} = L(f_\theta(x), y) + \lambda L(f_\theta(x + \delta), y)
```

**Python Example (Adversarial Training Loop):**
```python
for x, y in dataloader:
    x_adv = pgd_attack(model, x, y)
    output_clean = model(x)
    output_adv = model(x_adv)
    loss = criterion(output_clean, y) + criterion(output_adv, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 7. References
- [FGSM Paper](https://arxiv.org/abs/1412.6572)
- [PGD Paper](https://arxiv.org/abs/1706.06083)
- [C&W Paper](https://arxiv.org/abs/1608.04644)
- [Adversarial Training Paper](https://arxiv.org/abs/1706.06083) 