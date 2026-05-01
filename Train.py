import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate, add, Multiply, Lambda
from keras.models import Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from scipy.ndimage import gaussian_filter
from pathlib import Path
import os
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.abspath("Models_Arch"))
from mask_loss import MaskLoss
from mask_loss import MAELoss
# print(str(Path(__file__).parent))
from Models_Arch.ResidualUnet import Residual_Unet
from Models_Arch.Unet import Unet
from Models_Arch.Unet_7Kernel import Unet_7Kernel
from Models_Arch.Unet_5Kernel import Unet_5Kernel
from Models_Arch.Unet_3Dense import Unet_3Dense
from Models_Arch.Unet_1Dense import Unet_1Dense
from Models_Arch.Unet_2Dense import Unet_2Dense
from Models_Arch.Unet_1Dense_7Kernel import Unet_1Dense_7Kernel
from Models_Arch.Unet_1Dense_5Kernel import Unet_1Dense_5Kernel
from Models_Arch.Unet_2Dense_7Kernel import Unet_2Dense_7Kernel
from Models_Arch.Unet_2Dense_5Kernel import Unet_2Dense_5Kernel
from Models_Arch.Unet_3Dense_7Kernel import Unet_3Dense_7Kernel
from Models_Arch.Unet_3Dense_5Kernel import Unet_3Dense_5Kernel
from Models_Arch.ResidualUnet_1Dense import Residual_Unet_1D
from Models_Arch.ResidualUnet_2Dense import Residual_Unet_2D
from Models_Arch.ResidualUnet_3Dense import Residual_Unet_3D
from Models_Arch.ResidualUnet_1Dense_7Kernels import Residual_Unet_1D_7K
from Models_Arch.ResidualUnet_1Dense_5Kernels import Residual_Unet_1D_5K
from Models_Arch.ResidualUnet_2Dense_7Kernels import Residual_Unet_2D_7K
from Models_Arch.ResidualUner_2Dense_5Kernels import Residual_Unet_2D_5K
from Models_Arch.ResidualUnet_3Dense_7Kernels import Residual_Unet_3D_7K
from Models_Arch.ResidualUnet_3Dense_5Kernels import Residual_Unet_3D_5K
from Converter import Converter

class Train:
    def __init__(self, dataset_path, models_list, save_dir, saved_model_dir):
        """
        Initialize the Automate_Training class with paths and model configurations.

        Args:
            dataset_path (str): Path to the dataset containing 'Displacement' and 'Frames' directories.
            real_test_data_path (str): Path to real test data (e.g., patient data).
            models_list (list): List of models to be trained.
            save_dir (str): Directory to save visualization outputs and results.
            saved_model_dir (str): Directory to save trained model checkpoints.
        """
        self.dataset_path = dataset_path
        self.models_list = models_list
        self.save_dir = save_dir
        self.saved_model_dir = saved_model_dir
        self.first_images_train = None
        self.first_images_test = None
        self.second_images_train = None
        self.second_images_test = None
        self.disparity_train = None
        self.disparity_test = None
        self.original_flag = True
        self.all_losses = []
        self.all_val_losses = []
        self.model_names = []
        self.converter = Converter(focal_length=768.16058349609375)
        self.load_data()

    def load_data(self):
        for folder in os.listdir(self.dataset_path):
            for file in os.listdir(os.path.join(self.dataset_path, folder)):
                if "left.jpg" in file:
                    if self.first_images_train is None:
                        self.first_images_train = []
                    self.first_images_train.append(os.path.join(self.dataset_path, folder, file))
                elif "right.jpg" in file:
                    if self.second_images_train is None:
                        self.second_images_train = []
                    self.second_images_train.append(os.path.join(self.dataset_path, folder, file))
                elif "left.depth.png" in file:
                    if self.disparity_train is None:
                        self.disparity_train = []
                    self.converter.Convert_Depth_to_Disparity(os.path.join(self.dataset_path, folder, file), baseline=6.0)
                    
