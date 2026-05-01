import numpy as np
import matplotlib.pyplot as plt

class plotter:
    def __init__(self, file_path):
        self.file_path = file_path

    def plot_disparity_map(self):
        disparity = np.load(self.file_path)

        plt.imshow(disparity, cmap="plasma")
        plt.colorbar(label="Disparity")
        plt.title("Disparity Map")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.show()

    def plot_point_cloud(self, point_cloud):
        pc = np.load(
            self.file_path
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

# Example usage:
if __name__ == "__main__":
    # Change this to your file path
    file_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000000.left.depth_disparity.npy"
    
    plotter_instance = plotter(file_path)
    plotter_instance.plot_disparity_map()