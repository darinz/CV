# Applications of 3D Vision

3D vision techniques are foundational in many real-world applications, enabling machines to perceive, understand, and interact with the three-dimensional world. Here are some of the most impactful applications:

---

## 1. 3D Object Recognition

**3D object recognition** involves identifying and classifying objects from 3D data (point clouds, meshes, voxels, etc.). This is crucial for robotics, autonomous vehicles, and industrial automation.

### Example: Point Cloud Classification with a Simple MLP

```python
import torch
import torch.nn as nn

class PointNetLike(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU()
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        # x: (B, N, 3) - B: batch, N: points
        x = self.mlp(x)  # (B, N, 256)
        x = x.max(dim=1)[0]  # Global max pooling (B, 256)
        return self.fc(x)

# Example usage:
# model = PointNetLike(num_classes=5)
# pc = torch.randn(8, 1024, 3)  # 8 point clouds, 1024 points each
# logits = model(pc)
```

---

## 2. Scene Reconstruction

**Scene reconstruction** builds a 3D model of an environment from images or sensor data. This is used in mapping, AR/VR, and digital twins.

### Example: Visualizing a Reconstructed Scene from Depth Maps

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulate a depth map (e.g., from a depth camera)
H, W = 64, 64
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
X, Y = np.meshgrid(x, y)
Z = np.exp(-X**2 - Y**2)  # Example surface

# Convert depth map to 3D points
points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
ax.set_title('3D Scene Reconstruction from Depth Map')
plt.show()
```

---

## 3. Robotics and Navigation

Robots use 3D vision for perception, mapping, and navigation in complex environments. Tasks include obstacle avoidance, SLAM (Simultaneous Localization and Mapping), and grasp planning.

### Example: Obstacle Detection from Point Clouds (Pseudocode)

```python
# Given a point cloud 'pc' (N, 3) and robot's current position
# Detect obstacles within a certain radius
radius = 0.5  # meters
robot_pos = np.array([0, 0, 0])
distances = np.linalg.norm(pc - robot_pos, axis=1)
obstacles = pc[distances < radius]
print(f"Detected {len(obstacles)} obstacles within {radius}m")
```

### SLAM Overview
- **SLAM** combines 3D vision (feature extraction, depth estimation) with probabilistic mapping to build a map and localize the robot.
- Libraries: [RTAB-Map](https://introlab.github.io/rtabmap/), [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2)

---

## 4. AR/VR Content Creation

3D vision enables the creation and manipulation of virtual objects and environments for Augmented Reality (AR) and Virtual Reality (VR).

### Example: Placing a Virtual Object Using Camera Pose and Depth

```python
# Given camera pose (R, t) and a depth map, place a virtual cube at a real-world location
# (Pseudocode for concept)
virtual_object_pos = np.array([0.5, 0.2, 1.0])  # In world coordinates
# Project to image using camera intrinsics and pose
# Render the cube at the projected location in the AR scene
```

- Libraries: [Open3D](http://www.open3d.org/), [ARCore](https://developers.google.com/ar), [ARKit](https://developer.apple.com/augmented-reality/)

---

## Summary Table

| Application            | Description                                 | Example Libraries/Tools         |
|-----------------------|---------------------------------------------|---------------------------------|
| 3D Object Recognition | Classify objects from 3D data               | PyTorch3D, Open3D, PointNet     |
| Scene Reconstruction  | Build 3D models from images/sensors         | Open3D, COLMAP, Meshroom        |
| Robotics/Navigation   | Perception, mapping, obstacle avoidance     | ROS, RTAB-Map, ORB-SLAM         |
| AR/VR Content         | Virtual object creation and interaction     | ARCore, ARKit, Open3D           |

---

3D vision is a rapidly growing field with applications across robotics, entertainment, industry, and science. Mastery of these techniques opens the door to building intelligent systems that understand and interact with the 3D world. 