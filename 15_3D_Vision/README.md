# 3D Vision

3D vision studies the representation, reconstruction, and understanding of three-dimensional shapes and scenes from images, point clouds, or other sensor data. This field is foundational for robotics, AR/VR, autonomous driving, and more.

## 3D Shape Representations

### Point Clouds
A point cloud is a set of 3D points $`\{\mathbf{p}_i\}_{i=1}^N`$, where each $`\mathbf{p}_i \in \mathbb{R}^3`$.

- **Advantages:** Simple, flexible, directly acquired from sensors (e.g., LiDAR).
- **Challenges:** No explicit connectivity, unordered, varying density.

### Meshes
A mesh represents a surface as a set of vertices $`V`$ and faces $`F`$:
```math
M = (V, F)
```
where $`V = \{\mathbf{v}_i\}_{i=1}^N`$, $`\mathbf{v}_i \in \mathbb{R}^3`$, and $`F`$ defines connectivity (typically triangles).

- **Advantages:** Encodes surface connectivity, widely used in graphics.
- **Challenges:** Fixed topology, complex to process.

### Voxels
A voxel grid discretizes 3D space into a regular grid:
```math
V \in \{0,1\}^{X \times Y \times Z}
```
where $`V_{x,y,z}`$ indicates occupancy.

- **Advantages:** Regular structure, easy for 3D CNNs.
- **Challenges:** High memory cost, limited resolution.

### Signed Distance Functions (SDF)
An SDF represents a shape as a function $`f: \mathbb{R}^3 \to \mathbb{R}`$:
```math
f(\mathbf{x}) = \begin{cases}
< 0 & \text{inside surface} \\
= 0 & \text{on surface} \\
> 0 & \text{outside surface}
\end{cases}
```

## Shape Reconstruction

Shape reconstruction aims to recover 3D geometry from 2D images, depth maps, or partial 3D data.

### Multi-View Stereo (MVS)
Given $`N`$ images $`\{I_i\}`$ with known camera poses, reconstruct a 3D shape by finding correspondences and triangulating points.

#### Triangulation
Given two camera matrices $`P_1, P_2`$ and corresponding image points $`x_1, x_2`$:
```math
\lambda_1 x_1 = P_1 X, \quad \lambda_2 x_2 = P_2 X
```
Solve for 3D point $`X`$.

### Depth Estimation
Estimate per-pixel depth $`d(u, v)`$ from a single image $`I`$:
```math
D = f(I)
```
where $`D \in \mathbb{R}^{H \times W}`$.

### Volumetric Reconstruction
Predict a voxel grid $`V`$ from input data:
```math
V = f(I)
```
where $`f`$ is a neural network (e.g., 3D CNN).

### Surface Reconstruction from Point Clouds
Fit a surface to a set of points $`\{\mathbf{p}_i\}`$ using algorithms like Poisson surface reconstruction or neural networks.

## Neural Implicit Representations

Neural implicit representations model 3D shapes as continuous functions parameterized by neural networks.

### Occupancy Networks
Learn a function $`f_\theta: \mathbb{R}^3 \to [0,1]`$ that predicts occupancy:
```math
\hat{o} = f_\theta(\mathbf{x})
```
where $`\hat{o}`$ is the probability that $`\mathbf{x}`$ is inside the object.

### Deep Signed Distance Functions (DeepSDF)
Learn a neural network $`f_\theta: \mathbb{R}^3 \to \mathbb{R}`$ to predict the signed distance:
```math
\hat{s} = f_\theta(\mathbf{x})
```
where $`\hat{s}`$ is the signed distance at $`\mathbf{x}`$.

### Neural Radiance Fields (NeRF)
Model a scene as a function $`F_\theta(\mathbf{x}, \mathbf{d}) = (c, \sigma)`$ where $`\mathbf{x}`$ is a 3D position, $`\mathbf{d}`$ is a viewing direction, $`c`$ is color, and $`\sigma`$ is volume density:
```math
(c, \sigma) = F_\theta(\mathbf{x}, \mathbf{d})
```
Rendering is performed by integrating along camera rays:
```math
C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), \mathbf{d}) dt
```
where $`T(t)`$ is the accumulated transmittance.

## Applications
- 3D object recognition
- Scene reconstruction
- Robotics and navigation
- AR/VR content creation

## Summary

3D vision leverages a variety of shape representations and reconstruction techniques. Neural implicit representations have enabled high-fidelity, memory-efficient modeling of complex 3D geometry, revolutionizing the field and enabling new applications in graphics, robotics, and beyond. 