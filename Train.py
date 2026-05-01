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
from Plotter import plotter

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

# import tensorflow as tf

# print("TensorFlow version:", tf.__version__)
# print("GPUs:", tf.config.list_physical_devices("GPU"))
# print("All devices:", tf.config.list_physical_devices())

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
        self.first_images = None
        self.second_images = None
        self.disparity_maps = None
        self.first_images_train = None
        self.first_images_test = None
        self.first_images_val = None
        self.second_images_train = None
        self.second_images_test = None
        self.second_images_val = None
        self.disparity_train = None
        self.disparity_test = None
        self.disparity_val = None
        self.original_flag = True
        self.all_losses = []
        self.all_val_losses = []
        self.model_names = []
        self.converter = Converter(focal_length=768.16058349609375)
        self.load_data()

    def load_data(self):
        # Orde
        for folder in os.listdir(self.dataset_path):
            if folder == "kitchen_0":
                # Order the files in the folder to ensure left and right images are matched correctly
                files = sorted(os.listdir(os.path.join(self.dataset_path, folder)))
                for file in files:
                    if "left.jpg" in file:
                        if self.first_images is None:
                            self.first_images = []
                        print(f"Loading first image: {file}")
                        first_image = cv2.imread(os.path.join(self.dataset_path, folder, file))
                        #resize to 256, 480
                        first_image = cv2.resize(first_image, (480, 256))
                        self.first_images.append(first_image)
                    elif "right.jpg" in file:
                        if self.second_images is None:
                            self.second_images = []
                        second_image = cv2.imread(os.path.join(self.dataset_path, folder, file))
                        #resize to 512, 960
                        second_image = cv2.resize(second_image, (480, 256))
                        self.second_images.append(second_image)
                    elif "left.depth.png" in file:
                        if self.disparity_maps is None:
                            self.disparity_maps = []
                        disparity = self.converter.Convert_Depth_to_Disparity(os.path.join(self.dataset_path, folder, file), baseline=6.0)
                        disparity = cv2.resize(disparity, (480, 256))
                        self.disparity_maps.append(disparity)

        self.first_images_train = np.array(self.first_images[:int(0.7 * len(self.first_images))])
        self.first_images_valid = np.array(self.first_images[int(0.7 * len(self.first_images)):int(0.85 * len(self.first_images))])
        self.first_images_test = np.array(self.first_images[int(0.85 * len(self.first_images)):])
        self.second_images_train = np.array(self.second_images[:int(0.7 * len(self.second_images))])
        self.second_images_valid = np.array(self.second_images[int(0.7 * len(self.second_images)):int(0.85 * len(self.second_images))])
        self.second_images_test = np.array(self.second_images[int(0.85 * len(self.second_images)):])
        self.disparity_train = np.array(self.disparity_maps[:int(0.7 * len(self.disparity_maps))])
        self.disparity_valid = np.array(self.disparity_maps[int(0.7 * len(self.disparity_maps)):int(0.85 * len(self.disparity_maps))])
        self.disparity_test = np.array(self.disparity_maps[int(0.85 * len(self.disparity_maps)):])  
        # #plot the first image, second image and disparity map to check they are loaded correctly using matplotlib
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(cv2.cvtColor(self.first_images_train[0], cv2.COLOR_BGR2RGB))
        # axs[0].set_title("First Image")
        # axs[0].axis("off")
        # axs[1].imshow(cv2.cvtColor(self.second_images_train[0], cv2.COLOR_BGR2RGB))
        # axs[1].set_title("Second Image")
        # axs[1].axis("off")
        # im = axs[2].imshow(self.disparity_train[0], cmap="plasma")
        # axs[2].set_title("Disparity Map")
        # axs[2].axis("off")
        # fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
        # plt.show()

    def train_models(self, batch_size=2, num_epochs=10):
        # Placeholder for training logic
        for model in self.models_list:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
            first_image = Input(shape=(256, 480, 3), name="first_image")
            second_image = Input(shape=(256, 480, 3), name="second_image")
            modelnet = model
            model_name = model.__class__.__name__
            self.model_names.append(model_name)
            out_def = modelnet([first_image, second_image])
            model = Model(inputs=[first_image, second_image], outputs=out_def)

            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            file_name = f"{model_name}.keras"  
            check_point_path = os.path.join(self.saved_model_dir, file_name)
            # Create an empty .keras file
            open(check_point_path, "w").close()


            # Define callbacks
            checkpoint_callback = ModelCheckpoint(
                check_point_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1

            )

            stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min',
                verbose=1
            )
            with tf.device('/GPU:0'):
                history = model.fit(
                    [self.first_images_train, self.second_images_train],
                    self.disparity_train,
                    batch_size=batch_size,
                    epochs = num_epochs,
                    validation_data=([self.first_images_valid, self.second_images_valid], self.disparity_valid),
                    callbacks=[checkpoint_callback, stopping_callback],
                )
            #Create folder for the model
            self.visualise_outputs(model_name, history, model)
            print(f"Model {model_name} has been trained successfully")

# Example usage:
if __name__ == "__main__":
    dataset_path = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/Dataset/NVIDIA/fat/mixed"
    models_list = [Unet()]  # Add more model names as needed
    save_dir = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/visuals"
    saved_model_dir = "/Users/ahmed_ali/Documents/GitHub/DeepView3D/saved_models"

    trainer = Train(dataset_path, models_list, save_dir, saved_model_dir)
    trainer.train_models(batch_size=2, num_epochs=10)
