# 3D Shape Representations

Understanding how to represent 3D shapes is fundamental in computer vision, graphics, and robotics. Here, we cover the four most common representations: Point Clouds, Meshes, Voxels, and Signed Distance Functions (SDFs).

---

## 1. Point Clouds

A **point cloud** is a set of 3D points $\{\mathbf{p}_i\}_{i=1}^N$, where each $\mathbf{p}_i \in \mathbb{R}^3$.

- **Advantages:** Simple, flexible, directly acquired from sensors (e.g., LiDAR, depth cameras).
- **Challenges:** No explicit connectivity, unordered, varying density.

### Example: Visualizing a Point Cloud

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random 3D points (e.g., simulating a LiDAR scan)
N = 1000
points = np.random.uniform(-1, 1, (N, 3))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
ax.set_title('Random 3D Point Cloud')
plt.show()
```

---

## 2. Meshes

A **mesh** represents a surface as a set of vertices $V$ and faces $F$:

$$
M = (V, F)
$$

- $V = \{\mathbf{v}_i\}_{i=1}^N$, $\mathbf{v}_i \in \mathbb{R}^3$
- $F$ defines connectivity (typically triangles)

- **Advantages:** Encodes surface connectivity, widely used in graphics.
- **Challenges:** Fixed topology, more complex to process.

### Example: Creating and Visualizing a Simple Mesh

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Vertices of a cube
V = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])
# Faces (each as a list of vertex indices)
F = [
    [0, 1, 2, 3], [4, 5, 6, 7], # top and bottom
    [0, 1, 5, 4], [2, 3, 7, 6], # sides
    [1, 2, 6, 5], [0, 3, 7, 4]
]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection([V[face] for face in F], alpha=0.5, facecolor='cyan')
ax.add_collection3d(mesh)
ax.scatter(V[:, 0], V[:, 1], V[:, 2], color='k')
ax.set_title('Cube Mesh')
plt.show()
```

---

## 3. Voxels

A **voxel grid** discretizes 3D space into a regular grid:

$$
V \in \{0,1\}^{X \times Y \times Z}
$$

- $V_{x,y,z}$ indicates occupancy (1 = occupied, 0 = empty)
- **Advantages:** Regular structure, easy for 3D CNNs
- **Challenges:** High memory cost, limited resolution

### Example: Visualizing a Voxel Grid (3D Binary Array)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D grid with a filled sphere
X, Y, Z = 32, 32, 32
x, y, z = np.indices((X, Y, Z))
center = np.array([16, 16, 16])
radius = 10
voxels = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) < radius**2

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(voxels, edgecolor='k')
ax.set_title('Voxelized Sphere')
plt.show()
```

---

## 4. Signed Distance Functions (SDF)

A **Signed Distance Function** represents a shape as a function $f: \mathbb{R}^3 \to \mathbb{R}$:

$$
f(\mathbf{x}) = \begin{cases}
< 0 & \text{inside surface} \\
= 0 & \text{on surface} \\
> 0 & \text{outside surface}
\end{cases}
$$

- **Advantages:** Continuous, can represent complex shapes, used in implicit neural representations
- **Challenges:** Harder to visualize directly, requires function evaluation

### Example: SDF for a Sphere

```python
import numpy as np
import matplotlib.pyplot as plt

# SDF for a sphere of radius r centered at c
def sdf_sphere(x, y, z, c=(0,0,0), r=1.0):
    return np.sqrt((x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2) - r

# Visualize SDF slice (z=0)
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = 0
SDF = sdf_sphere(X, Y, Z)

plt.contourf(X, Y, SDF, levels=50, cmap='RdBu')
plt.colorbar(label='Signed Distance')
plt.contour(X, Y, SDF, levels=[0], colors='k', linewidths=2)
plt.title('SDF Slice for a Sphere (z=0)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

---

## Summary Table

| Representation | Structure | Pros | Cons |
|----------------|-----------|------|------|
| Point Cloud    | Set of 3D points | Simple, sensor-friendly | No connectivity, unordered |
| Mesh           | Vertices + Faces | Surface connectivity | Fixed topology |
| Voxel Grid     | 3D array         | Regular, 3D CNNs     | Memory cost |
| SDF            | Function         | Continuous, flexible | Harder to visualize |

---

These representations form the basis for most 3D vision and graphics pipelines. Later sections will show how to reconstruct, process, and learn from these representations. 