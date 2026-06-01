# DeepView3D

---

# Deep Learning

This section covers the deep-learning-based stereo disparity estimation pipeline located in `DeepLearningApproach/`.

## Overview

The pipeline takes pairs of stereo images (left + right) and predicts a disparity map, which can then be converted to depth or a 3D point cloud. The workflow has three stages:

1. **Data preparation** — convert raw images/depth files to `.npy` arrays (`main.py`)
2. **Training** — train one or more U-Net variants (`Train.py`)
3. **Testing / inference** — evaluate a saved model on held-out data (`Test.py`)

---

## Folder Structure

```
DeepLearningApproach/
├── main.py                  # Data preparation: raw dataset → .npy splits
├── Train.py                 # Training loop for all model variants
├── Test.py                  # Inference and visual evaluation
├── Loss.py                  # Custom masked Huber loss (MaskedLoss)
├── SOT.ipynb                                              # SOT model notebook
├── SOT_SOTA_Comparison.ipynb                             # SOT vs SOTA visual comparison
├── SOT_SOTA_Comparison_npy.ipynb                         # SOT vs SOTA on .npy data
├── SOT_SOTA_Comparison_npy_colorbar.ipynb                # Above with colorbar
├── Residual_SOTA_Comparison_test_npy.ipynb               # Residual vs SOTA on test .npy
├── Residual_SOTA_Comparison_kitedemo_single.ipynb        # Residual vs SOTA on kite demo
├── Residual_SOTA_Comparison_kitedemo_single_zoom.ipynb   # Above with zoom
├── Residual_SOTA_Comparison_kitedemo_single_zoom_npy.ipynb # Above using .npy input
├── Residual_SOTA_Comparison_real_sample.ipynb            # Residual vs SOTA on real sample
└── Models_Arch/             # All model architecture definitions
    ├── Unet.py
    ├── ResidualUnet.py
    ├── Unet_{1,2,3}Dense*.py
    ├── ResidualUnet_{1,2,3}Dense*.py
    └── ...                  # Variants with 5- and 7-pixel kernels, CoordConv, etc.
```

---

## Dataset Structure

Each scene folder inside your dataset root must contain three files per frame:

```
Dataset/
└── <scene_name>/
    ├── 000000.left.jpg         # Left stereo frame
    ├── 000000.right.jpg        # Right stereo frame
    └── 000000.left.depth.png   # Depth map for the left frame
```

All images are resized to **480 × 256** during loading.

---

## Step 1 — Prepare the Data (`main.py`)

`main.py` walks the raw dataset, loads every matching file triplet, resizes them, splits them 70 / 15 / 15 (train / validation / test), and saves nine `.npy` files.

**Edit the two path variables at the bottom of `main.py`:**

```python
# main.py — bottom of file

dataset_path = "/path/to/your/Dataset"   # folder that contains scene sub-folders
output_dir   = "/path/to/your/Dataset"   # where the .npy files will be written
```

Run it:

```bash
cd DeepLearningApproach
python main.py
```

This produces in `output_dir`:

| File | Contents |
|---|---|
| `first_images_train.npy` | Left frames — training split |
| `first_images_valid.npy` | Left frames — validation split |
| `first_images_test.npy` | Left frames — test split |
| `second_images_train.npy` | Right frames — training split |
| `second_images_valid.npy` | Right frames — validation split |
| `second_images_test.npy` | Right frames — test split |
| `disparity_train.npy` | Disparity maps — training split |
| `disparity_valid.npy` | Disparity maps — validation split |
| `disparity_test.npy` | Disparity maps — test split |

---

## Step 2 — Train (`Train.py`)

### Paths to update

Open `Train.py` and update the nine `np.load(...)` calls inside `__init__` to point to the `.npy` files you just created:

```python
# Train.py — inside Train.__init__

self.first_images_train  = np.load("/path/to/your/Dataset/first_images_train.npy")
self.first_images_valid  = np.load("/path/to/your/Dataset/first_images_valid.npy")
self.first_images_test   = np.load("/path/to/your/Dataset/first_images_test.npy")
self.second_images_train = np.load("/path/to/your/Dataset/second_images_train.npy")
self.second_images_valid = np.load("/path/to/your/Dataset/second_images_valid.npy")
self.second_images_test  = np.load("/path/to/your/Dataset/second_images_test.npy")
self.disparity_train     = np.load("/path/to/your/Dataset/disparity_train.npy")
self.disparity_valid     = np.load("/path/to/your/Dataset/disparity_valid.npy")
self.disparity_test      = np.load("/path/to/your/Dataset/disparity_test.npy")
```

Also update the `__main__` block at the bottom:

```python
# Train.py — __main__ block

dataset_path    = "/path/to/your/Dataset"           # only used if calling load_data()
save_dir        = "/path/to/save/visuals"            # loss plots, sample outputs
saved_model_dir = "/path/to/save/saved_models"       # .keras checkpoint files
```

### Choosing which models to train

Add any combination of the available architectures to `models_list`:

```python
from Models_Arch.Unet import Unet
from Models_Arch.ResidualUnet import Residual_Unet
from Models_Arch.ResidualUnet_3Dense_5Kernels import Residual_Unet_3D_5K
# ... etc.

models_list = [
    Unet(),
    Residual_Unet(),
    Residual_Unet_3D_5K(),
]
```

### Running training

```bash
cd DeepLearningApproach
python Train.py
```

Training runs each model sequentially with:
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5, clipnorm=1.0)
- **Loss**: `MaskedLoss` — masked smooth-L1 that ignores zero/invalid disparity pixels
- **Callbacks**: `ModelCheckpoint` (saves best val_loss) + `EarlyStopping` (patience=7)
- **Default**: batch size 2, 70 epochs

To change batch size or epoch count:

```python
trainer.train_models(batch_size=4, num_epochs=100)
```

### Outputs

For each model `<ModelName>`, the following are written to `save_dir/<ModelName>/`:

| File | Description |
|---|---|
| `loss_plot<ModelName>.png` | Training vs. validation loss curve |
| `sample_output<ModelName>.png` | Predicted vs. ground-truth disparity for one sample |
| `evaluation_results<ModelName>.txt` | Test loss and MAE |

Loss arrays are also saved at the root of `save_dir` as `losses<ModelName>.npy` and `val_losses<ModelName>.npy`.

Trained model weights are saved to `saved_model_dir/<ModelName>.keras`.

---

## Step 3 — Test / Inference (`Test.py`)

### Paths to update

```python
# Test.py — inside Test.__init__

self.first_image_train  = np.load("/path/to/your/Dataset/first_images_valid.npy")
self.second_image_train = np.load("/path/to/your/Dataset/second_images_valid.npy")
self.disparity_train    = np.load("/path/to/your/Dataset/disparity_valid.npy")
```

```python
# Test.py — inside Test.load_model()

model = tf.keras.models.load_model(
    "/path/to/saved_models/Residual_Unet_3D_5K.keras",
    custom_objects={'Residual_Unet_3D_5K': Residual_Unet_3D_5K, 'MaskedLoss': MaskedLoss}
)
```

```python
# Test.py — __main__ block

model_path        = "/path/to/saved_models/Residual_Unet_3D_5K.keras"
right_image_path  = "/path/to/Dataset/scene/000000.right.jpg"
left_image_path   = "/path/to/Dataset/scene/000000.left.jpg"
disparity_map_path = "/path/to/Dataset/scene/000000.left.depth.png"
```

### Running the test

```bash
cd DeepLearningApproach
python Test.py
```

This runs inference on up to 500 validation samples (50 per page) and saves paginated PNG grids:

```
test_500_disparity_results_page_1.png
test_500_disparity_results_page_2.png
...
```

Each grid shows: **Left | Right | Predicted | Ground Truth** for every sample.

---

## Available Model Architectures

All models live in `Models_Arch/` and accept two inputs `[left_image, right_image]` of shape `(256, 480, 3)`.

| Class | File | Description |
|---|---|---|
| `Unet` | `Unet.py` | Baseline U-Net, 3×3 kernels |
| `Unet_5Kernel` | `Unet_5Kernel.py` | U-Net with 5×5 kernels |
| `Unet_7Kernel` | `Unet_7Kernel.py` | U-Net with 7×7 kernels |
| `Unet_1Dense` | `Unet_1Dense.py` | U-Net + 1 dense block |
| `Unet_2Dense` | `Unet_2Dense.py` | U-Net + 2 dense blocks |
| `Unet_3Dense` | `Unet_3Dense.py` | U-Net + 3 dense blocks |
| `Unet_{1,2,3}Dense_{5,7}Kernel` | respective files | Dense U-Net variants with larger kernels |
| `Residual_Unet` | `ResidualUnet.py` | Residual U-Net, 3×3 kernels |
| `Residual_Unet_1D` | `ResidualUnet_1Dense.py` | Residual U-Net + 1 dense block |
| `Residual_Unet_2D` | `ResidualUnet_2Dense.py` | Residual U-Net + 2 dense blocks |
| `Residual_Unet_3D` | `ResidualUnet_3Dense.py` | Residual U-Net + 3 dense blocks |
| `Residual_Unet_3D_5K` | `ResidualUnet_3Dense_5Kernels.py` | Residual 3-dense with 5×5 kernels (best performing) |
| `Residual_Unet_3D_5K_CoordConv` | `ResidualUnet_3Dense_5Kernels_CoordConv.py` | Above + coordinate convolutions |
| `Residual_Unet_3D_5K_FeatureExtractor` | `ResidualUnet_3Dense_5Kernels_FeatureExtractor.py` | Above + learned feature extractor |

---

## Notebooks

The following Jupyter notebooks live in `DeepLearningApproach/` and cover model evaluation and visual comparison against SOTA methods:

| Notebook | Description |
|---|---|
| `SOT.ipynb` | SOT model exploration and evaluation |
| `SOT_SOTA_Comparison.ipynb` | Visual comparison of SOT vs SOTA on image inputs |
| `SOT_SOTA_Comparison_npy.ipynb` | Same comparison using `.npy` data |
| `SOT_SOTA_Comparison_npy_colorbar.ipynb` | Above with colorbar overlays |
| `Residual_SOTA_Comparison_test_npy.ipynb` | Residual U-Net vs SOTA on test `.npy` data |
| `Residual_SOTA_Comparison_kitedemo_single.ipynb` | Residual vs SOTA on a single kite-demo image |
| `Residual_SOTA_Comparison_kitedemo_single_zoom.ipynb` | Above with zoomed-in view |
| `Residual_SOTA_Comparison_kitedemo_single_zoom_npy.ipynb` | Above using `.npy` input |
| `Residual_SOTA_Comparison_real_sample.ipynb` | Residual vs SOTA on a real-world sample |

---

## Dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib scipy pillow
```

GPU training is strongly recommended. The training loop pins computation to `/GPU:0` automatically when a GPU is available.