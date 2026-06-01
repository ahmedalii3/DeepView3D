import os
import cv2
import numpy as np
from Converter import Converter

def load_and_save_data(dataset_path,  output_dir):
    """
    Load left/right images and disparity maps from dataset_path, split into train/validation/test,
    and save them as .npy files in output_dir.
    
    Args:
        dataset_path (str): Path to dataset containing folders of images.
        converter: An object with a method Convert_Depth_to_Disparity(file_path, baseline).
        output_dir (str): Directory to save the .npy files.
    """
    converter = Converter(focal_length=768.16058349609375)
    first_images = []
    second_images = []
    disparity_maps = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through folders in dataset
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # Order files in the folder
        files = sorted(os.listdir(folder_path))
        for file in files:
            file_path = os.path.join(folder_path, file)

            if "left.jpg" in file:
                print(f"Loading first image: {file}")
                img = cv2.imread(file_path)
                img = cv2.resize(img, (480, 256))
                first_images.append(img)

            elif "right.jpg" in file:
                img = cv2.imread(file_path)
                img = cv2.resize(img, (480, 256))
                second_images.append(img)

            elif "left.depth.png" in file:
                # disparity = converter.Convert_Depth_to_Disparity(file_path, baseline=6.0)
                disparity= cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                disparity = cv2.resize(disparity, (480, 256))
                disparity_maps.append(disparity)
    
    # Convert to numpy arrays
    first_images = np.array(first_images)
    second_images = np.array(second_images)
    disparity_maps = np.array(disparity_maps)

    # Compute split indices
    n = len(first_images)
    train_end = int(0.7 * n)
    valid_end = int(0.85 * n)

    # Split datasets
    first_images_train, first_images_valid, first_images_test = first_images[:train_end], first_images[train_end:valid_end], first_images[valid_end:]
    second_images_train, second_images_valid, second_images_test = second_images[:train_end], second_images[train_end:valid_end], second_images[valid_end:]
    disparity_train, disparity_valid, disparity_test = disparity_maps[:train_end], disparity_maps[train_end:valid_end], disparity_maps[valid_end:]

    # Save as .npy files
    np.save(os.path.join(output_dir, "first_images_train.npy"), first_images_train)
    np.save(os.path.join(output_dir, "first_images_valid.npy"), first_images_valid)
    np.save(os.path.join(output_dir, "first_images_test.npy"), first_images_test)
    np.save(os.path.join(output_dir, "second_images_train.npy"), second_images_train)
    np.save(os.path.join(output_dir, "second_images_valid.npy"), second_images_valid)
    np.save(os.path.join(output_dir, "second_images_test.npy"), second_images_test)
    np.save(os.path.join(output_dir, "disparity_train.npy"), disparity_train)
    np.save(os.path.join(output_dir, "disparity_valid.npy"), disparity_valid)
    np.save(os.path.join(output_dir, "disparity_test.npy"), disparity_test)

    print("Data successfully loaded and saved as .npy files.")



dataset_path= "/home/ha06508a/Downloads/DeepView3D-main/Dataset"

output_dir = "/home/ha06508a/Downloads/DeepView3D-main/Dataset"
load_and_save_data(dataset_path, output_dir)