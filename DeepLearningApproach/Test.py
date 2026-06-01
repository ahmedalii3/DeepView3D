
import cv2

from Models_Arch.Unet import Unet
from Models_Arch.ResidualUnet_3Dense_7Kernels import Residual_Unet_3D_7K
from Models_Arch.ResidualUnet_3Dense_5Kernels import Residual_Unet_3D_5K
from Loss import MaskedLoss
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from Models_Arch.Unet_5Kernel import Unet_5Kernel
from Converter import Converter
from Plotter import plotter

class Test:
    def __init__(self, model_path, model, right_image_path, left_image_path, disparity_map_path):
        self.model_path = model_path
        self.model = model
        self.right_image = right_image_path
        self.left_image = left_image_path
        self.disparity_map = disparity_map_path
        self.first_image_train = np.load("/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/first_images_valid.npy")
        self.second_image_train = np.load("/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/second_images_valid.npy")
        self.disparity_train = np.load("/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/disparity_valid copy.npy")
        self.converter = Converter()
        self.plotter = plotter(self.disparity_map)

    def load_model(self):
        model = tf.keras.models.load_model("/Users/ahmed_ali/Downloads/DeepView3D-main/Residual_Unet_3D_5K.keras", custom_objects={'Residual_Unet_3D_5K': Residual_Unet_3D_5K, 'MaskedLoss': MaskedLoss})
        return model
    
    # def test_model(self):
    #     # try 40 different n and see the results
    #     # last n 
        
    #     n = -1
    #     left_image = self.first_image_train[n]
    #     right_image = self.second_image_train[n]
    #     disparity_map = self.disparity_train[n]
    #     # plot the left and right images and the disparity map
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(left_image, cmap="gray")
    #     plt.title("Left Image")
    #     plt.xlabel("Pixel X")
    #     plt.ylabel("Pixel Y")

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(right_image, cmap="gray")
    #     plt.title("Right Image")
    #     plt.xlabel("Pixel X")
    #     plt.ylabel("Pixel Y")

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(disparity_map.squeeze(), cmap="plasma")
    #     plt.title("Disparity Map")
    #     plt.xlabel("Pixel X")
    #     plt.ylabel("Pixel Y")

    #     plt.show()

    #     print("Left image shape:", left_image.shape)
    #     print("Right image shape:", right_image.shape)
    #     print("Disparity map shape:", disparity_map.shape)

    #     model = self.load_model()

    #     left_batch = np.expand_dims(left_image, axis=0)
    #     right_batch = np.expand_dims(right_image, axis=0)

    #     print("Left batch shape:", left_batch.shape)
    #     print("Right batch shape:", right_batch.shape)

    #     output = model.predict([left_batch, right_batch]).squeeze()
    #     intrinsics = 	{
	# 			"fx": 768.16058349609375,
	# 			"fy": 768.16058349609375,
	# 			"cx": 480,
	# 			"cy": 270,
	# 			"s": 0
	# 	}
    #     original_w = 960

    #     original_h = 540

    #     new_h, new_w = output.shape
    #     print("new_h:", new_h)
    #     print("new_w:", new_w)

    #     scale_x = new_w / original_w    

    #     scale_y = new_h / original_h

    #     intrinsics_scaled = {

    #         "fx": intrinsics["fx"] * scale_x,

    #         "fy": intrinsics["fy"] * scale_y,

    #         "cx": intrinsics["cx"] * scale_x,

    #         "cy": intrinsics["cy"] * scale_y,

    #         "s": 0

    #     }
    #     # print the intrinsics after scaling
    #     print("Scaled intrinsics:", intrinsics_scaled)

    #     # save the output and disparity map
    #     np.save("ground_truth_depth_n.npy", disparity_map)
    #     np.save("predicted_depth_n.npy", output)
    #     gt = self.converter.convert_Depth_to_PointCloud(disparity_map,output_file="ground_truth_pointcloud.npy", intrinsics=intrinsics_scaled)
    #     pd = self.converter.convert_Depth_to_PointCloud(output, output_file="predicted_pointcloud.npy",intrinsics=intrinsics_scaled)

    #     self.plotter.plot_point_cloud(gt)
    #     self.plotter.plot_point_cloud(pd)


    #     print("Model output shape:", output.shape)
    #     print("Model output dtype:", output.dtype)

    #     plt.figure(figsize=(12, 6))

    #     plt.subplot(1, 2, 1)
    #     plt.imshow(output, cmap="plasma")
    #     plt.colorbar(label="Predicted Disparity")
    #     plt.title("Predicted Disparity Map")
    #     plt.xlabel("Pixel X")
    #     plt.ylabel("Pixel Y")

    #     plt.subplot(1, 2, 2)
    #     plt.imshow(disparity_map.squeeze(), cmap="plasma")
    #     plt.colorbar(label="Ground Truth Disparity")
    #     plt.title("Ground Truth Disparity Map")
    #     plt.xlabel("Pixel X")
    #     plt.ylabel("Pixel Y")

    #     plt.show()




    
    # def test_model(self):
    #     model = self.load_model()

    #     num_samples = 40
    #     start_n = 400

    #     plt.figure(figsize=(18, num_samples * 3))

    #     for i in range(num_samples):
    #         n = start_n + i

    #         left_image = self.first_image_train[n]
    #         right_image = self.second_image_train[n]
    #         disparity_map = self.disparity_train[n]

    #         left_batch = np.expand_dims(left_image, axis=0)
    #         right_batch = np.expand_dims(right_image, axis=0)

    #         output = model.predict([left_batch, right_batch], verbose=0).squeeze()

    #         plt.subplot(num_samples, 4, i * 4 + 1)
    #         plt.imshow(left_image.astype(np.uint8))
    #         plt.title(f"n={n} Left")
    #         plt.axis("off")

    #         plt.subplot(num_samples, 4, i * 4 + 2)
    #         plt.imshow(right_image.astype(np.uint8))
    #         plt.title("Right")
    #         plt.axis("off")

    #         plt.subplot(num_samples, 4, i * 4 + 3)
    #         plt.imshow(output, cmap="plasma")
    #         plt.title("Predicted")
    #         plt.axis("off")

    #         plt.subplot(num_samples, 4, i * 4 + 4)
    #         plt.imshow(disparity_map.squeeze(), cmap="plasma")
    #         plt.title("Ground Truth")
    #         plt.axis("off")

    #     plt.tight_layout()
    #     plt.savefig("test_40_disparity_results.png", dpi=200, bbox_inches="tight")
    #     plt.show()
    def test_model(self):
        model = self.load_model()

        num_samples = 500
        start_n = 0
        samples_per_page = 50

        total_samples = len(self.first_image_train)

        if start_n + num_samples > total_samples:
            raise ValueError(
                f"Requested samples from {start_n} to {start_n + num_samples - 1}, "
                f"but dataset only has {total_samples} samples."
            )

        for page_start in range(0, num_samples, samples_per_page):
            page_end = min(page_start + samples_per_page, num_samples)
            page_count = page_end - page_start

            plt.figure(figsize=(18, page_count * 3))

            for row, sample_offset in enumerate(range(page_start, page_end)):
                n = start_n + sample_offset

                left_image = self.first_image_train[n]
                right_image = self.second_image_train[n]
                disparity_map = self.disparity_train[n]

                left_batch = np.expand_dims(left_image, axis=0)
                right_batch = np.expand_dims(right_image, axis=0)

                output = model.predict([left_batch, right_batch], verbose=0).squeeze()

                plt.subplot(page_count, 4, row * 4 + 1)
                plt.imshow(left_image.astype(np.uint8))
                plt.title(f"n={n} Left")
                plt.axis("off")

                plt.subplot(page_count, 4, row * 4 + 2)
                plt.imshow(right_image.astype(np.uint8))
                plt.title("Right")
                plt.axis("off")

                plt.subplot(page_count, 4, row * 4 + 3)
                plt.imshow(output, cmap="plasma")
                plt.title("Predicted")
                plt.axis("off")

                plt.subplot(page_count, 4, row * 4 + 4)
                plt.imshow(disparity_map.squeeze(), cmap="plasma")
                plt.title("Ground Truth")
                plt.axis("off")

            plt.tight_layout()

            page_number = page_start // samples_per_page + 1
            output_name = f"test_500_disparity_results_page_{page_number}.png"

            plt.savefig(output_name, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Saved {output_name}")

if __name__ == "__main__":
    model_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/saved_models/Unet_5Kernel.keras"
    right_image_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_2/000000.left.jpg"
    left_image_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_2/000000.right.jpg"
    disparity_map_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_2/000000.left.depth.png"

    tester = Test(model_path, Unet_5Kernel(), right_image_path, left_image_path, disparity_map_path)
    tester.test_model()
        