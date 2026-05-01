
import cv2

from Models_Arch.Unet import Unet

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np


class Test:
    def __init__(self, model_path, model, right_image_path, left_image_path, disparity_map_path):
        self.model_path = model_path
        self.model = model
        self.right_image = right_image_path
        self.left_image = left_image_path
        self.disparity_map = disparity_map_path

    def load_model(self):
        model = tf.keras.models.load_model("saved_models/Unet.keras", custom_objects={'Unet': Unet})
        return model
    
    def test_model(self):
        self.right_image = cv2.imread(self.right_image)
        self.right_image = cv2.resize(self.right_image, (480, 256))
        self.left_image = cv2.imread(self.left_image)
        self.left_image = cv2.resize(self.left_image, (480, 256))
        # self.disparity_map = cv2.imread(self.disparity_map)
        self.disparity_map = np.load(self.disparity_map)
        self.disparity_map = cv2.resize(self.disparity_map, (480, 256))
        model = self.load_model()
        output = model.predict([self.left_image[None, ...], self.right_image[None, ...]]).squeeze()
        print("Model output shape:", output.shape)
        print("Model output dtype:", output.dtype)
        # plot the output disparity map beside the ground truth disparity map
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(output, cmap="plasma")
        plt.colorbar(label="Predicted Disparity")
        plt.title("Predicted Disparity Map")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.subplot(1, 2, 2)
        plt.imshow(self.disparity_map, cmap="plasma")
        plt.colorbar(label="Ground Truth Disparity")
        plt.title("Ground Truth Disparity Map")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.show()

if __name__ == "__main__":
    model_path = "saved_models/Unet.keras"
    right_image_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000001.right.jpg"
    left_image_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000001.left.jpg"
    disparity_map_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed/kitchen_0/000000.left.depth_disparity.npy"

    tester = Test(model_path, Unet(), right_image_path, left_image_path, disparity_map_path)
    tester.test_model()
        