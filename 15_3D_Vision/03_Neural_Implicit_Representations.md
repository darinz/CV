# Neural Implicit Representations

Neural implicit representations model 3D shapes or scenes as continuous functions parameterized by neural networks. These methods have enabled high-fidelity, memory-efficient modeling of complex 3D geometry.

---

## 1. Occupancy Networks

**Occupancy Networks** learn a function $f_\theta: \mathbb{R}^3 \to [0,1]$ that predicts whether a point $\mathbf{x}$ is inside an object (occupancy probability).

### Math
Given a 3D point $\mathbf{x}$, the network outputs $\hat{o} = f_\theta(\mathbf{x})$, where $\hat{o} \approx 1$ means inside, $\hat{o} \approx 0$ means outside.

### Example: Simple Occupancy Network in PyTorch

```python
import torch
import torch.nn as nn

class OccupancyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# Example usage:
# net = OccupancyNet()
# points = torch.randn(10, 3)  # 10 random 3D points
# occ = net(points)  # (10, 1), values in [0, 1]
```

---

## 2. Deep Signed Distance Functions (DeepSDF)

**DeepSDF** learns a neural network $f_\theta: \mathbb{R}^3 \to \mathbb{R}$ to predict the signed distance from a point to the surface of an object.

### Math
For a point $\mathbf{x}$:
- $f_\theta(\mathbf{x}) < 0$: inside the surface
- $f_\theta(\mathbf{x}) = 0$: on the surface
- $f_\theta(\mathbf{x}) > 0$: outside the surface

### Example: Simple DeepSDF Network in PyTorch

```python
import torch
import torch.nn as nn

class DeepSDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

# Example usage:
# net = DeepSDFNet()
# points = torch.randn(10, 3)
# sdf = net(points)  # (10, 1), signed distances
```

---

## 3. Neural Radiance Fields (NeRF)

**Neural Radiance Fields (NeRF)** model a scene as a function $F_\theta(\mathbf{x}, \mathbf{d}) = (c, \sigma)$, where:
- $\mathbf{x}$: 3D position
- $\mathbf{d}$: viewing direction
- $c$: color (RGB)
- $\sigma$: volume density

### Math
Rendering is performed by integrating along camera rays:

$$
C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), \mathbf{d}) dt
$$
where $T(t)$ is the accumulated transmittance.

### Example: Minimal NeRF MLP in PyTorch

```python
import torch
import torch.nn as nn

class NeRFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4)  # (R, G, B, sigma)
        )
    def forward(self, x, d):
        # x: (N, 3), d: (N, 3)
        inp = torch.cat([x, d], dim=-1)
        out = self.layers(inp)
        rgb = torch.sigmoid(out[:, :3])  # (N, 3)
        sigma = torch.relu(out[:, 3:4])  # (N, 1)
        return rgb, sigma

# Example usage:
# net = NeRFMLP()
# points = torch.randn(10, 3)
# dirs = torch.randn(10, 3)
# rgb, sigma = net(points, dirs)
```

---

## Summary Table

| Method            | Function Learned                | Output         | Use Case                  |
|-------------------|---------------------------------|---------------|---------------------------|
| Occupancy Network | $\mathbb{R}^3 \to [0,1]$        | Occupancy prob| Shape representation      |
| DeepSDF           | $\mathbb{R}^3 \to \mathbb{R}$   | Signed distance| Shape representation      |
| NeRF              | $\mathbb{R}^3, \mathbb{R}^3 \to \mathbb{R}^3, \mathbb{R}$ | Color, density | Scene rendering           |

---

Neural implicit representations are powerful tools for modeling complex 3D geometry and appearance, enabling applications in graphics, vision, and robotics. 