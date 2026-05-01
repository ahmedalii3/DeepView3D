import numpy as np
import matplotlib.pyplot as plt

# Load point cloud
pc = np.load(
    "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000000.left.depth_pointcloud.npy"
)

# Downsample for speed (optional)
pc = pc[::10]  # keep every 20th point

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Point Cloud")

plt.show()