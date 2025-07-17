# Shape Reconstruction

Shape reconstruction is the process of recovering 3D geometry from 2D images, depth maps, or partial 3D data. This guide covers the main approaches:

- Multi-View Stereo (MVS) & Triangulation
- Depth Estimation
- Volumetric Reconstruction
- Surface Reconstruction from Point Clouds

---

## 1. Multi-View Stereo (MVS) & Triangulation

**Multi-View Stereo (MVS)** reconstructs 3D shapes from multiple images taken from different viewpoints, given known camera poses. The core idea is to find correspondences between images and triangulate the 3D positions.

### Triangulation Math
Given two camera matrices $P_1, P_2$ and corresponding image points $x_1, x_2$:

$$
\lambda_1 x_1 = P_1 X, \quad \lambda_2 x_2 = P_2 X
$$

Solve for the 3D point $X$.

### Example: Triangulating a 3D Point from Two Views

```python
import numpy as np

def triangulate_point(P1, P2, x1, x2):
    """Triangulate a 3D point from two views."""
    A = np.array([
        x1[0]*P1[2,:] - P1[0,:],
        x1[1]*P1[2,:] - P1[1,:],
        x2[0]*P2[2,:] - P2[0,:],
        x2[1]*P2[2,:] - P2[1,:]
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]

# Example camera matrices (identity and translation)
P1 = np.hstack([np.eye(3), np.zeros((3,1))])
P2 = np.hstack([np.eye(3), np.array([[1],[0],[0]])])
# Image points (homogeneous)
x1 = np.array([0.5, 0.5, 1.0])
x2 = np.array([0.6, 0.5, 1.0])
X = triangulate_point(P1, P2, x1, x2)
print("Triangulated 3D point:", X)
```

---

## 2. Depth Estimation

**Depth estimation** predicts the distance from the camera to each pixel in an image, producing a depth map $D \in \mathbb{R}^{H \times W}$.

### Example: Visualizing a Synthetic Depth Map

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic depth map (e.g., a slanted plane)
H, W = 100, 100
x = np.linspace(0, 1, W)
y = np.linspace(0, 1, H)
X, Y = np.meshgrid(x, y)
D = 2.0 - X - 0.5*Y  # Depth decreases with X and Y

plt.imshow(D, cmap='plasma')
plt.colorbar(label='Depth')
plt.title('Synthetic Depth Map')
plt.xlabel('u')
plt.ylabel('v')
plt.show()
```

### Learning-based Depth Estimation
Modern methods use deep neural networks (e.g., CNNs) to estimate depth from a single image:

```python
import torch
import torch.nn as nn

class SimpleDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1), nn.ReLU()
        )
    def forward(self, x):
        x = self.encoder(x)
        d = self.decoder(x)
        return d  # Output: (B, 1, H, W)

# Example usage:
# model = SimpleDepthNet()
# image = torch.randn(1, 3, 128, 128)  # Dummy input
# depth = model(image)
```

---

## 3. Volumetric Reconstruction

**Volumetric reconstruction** predicts a voxel grid $V$ from input data (e.g., images or point clouds), often using 3D CNNs.

### Example: Predicting a Voxel Grid from an Image (Conceptual)

```python
import torch
import torch.nn as nn

class VoxelReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(8*32*32, 32*32*32)
    def forward(self, x):
        x = self.encoder(x)
        v = self.fc(x)
        v = v.view(-1, 32, 32, 32)
        return v  # Output: (B, 32, 32, 32)

# Example usage:
# model = VoxelReconstructor()
# image = torch.randn(1, 1, 32, 32)
# voxels = model(image)
```

---

## 4. Surface Reconstruction from Point Clouds

Given a set of points $\{\mathbf{p}_i\}$, surface reconstruction fits a continuous surface. Methods include Poisson surface reconstruction, alpha shapes, and neural networks.

### Example: Surface Reconstruction with Alpha Shapes (using open3d)

```python
import open3d as o3d
import numpy as np

# Generate random points on a sphere
points = np.random.randn(1000, 3)
points /= np.linalg.norm(points, axis=1, keepdims=True)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Alpha shape surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5)
o3d.visualization.draw_geometries([mesh])
```

---

## Summary

Shape reconstruction is a core problem in 3D vision, enabling the recovery of geometry from images and sensor data. Each method has its strengths and is suited to different scenarios, from multi-view setups to single-image inference and point cloud processing. 