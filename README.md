# DeepView3D

**Stereo-based 3D reconstruction comparing two paradigms on the same inputs:**
a **classical** Semi-Global Block Matching (SGBM) pipeline and a **deep-learning**
U-Net family that predicts dense depth from a stereo pair. Both produce a dense
depth map that is back-projected into a 3D point cloud, and both are evaluated on
the NVIDIA Falling Things (FAT) synthetic stereo dataset.

| Approach | Folder | Idea |
|---|---|---|
| **Classical** | [`ClassicalApproach/`](ClassicalApproach/) | Geometry-driven SGBM: rectify → disparity (with left–right fusion) → triangulate to metric depth → point cloud. No training. |
| **Deep Learning** | [`DeepLearningApproach/`](DeepLearningApproach/) | Custom U-Net family (21 variants) that takes left+right RGB and predicts dense disparity/depth, trained with a masked Huber loss. |

```
DeepView3D/
├── ClassicalApproach/        # Traditional SGBM stereo pipeline
│   ├── requirements.txt
│   └── sgbm/                  # Python package (run with `python -m sgbm...`)
├── DeepLearningApproach/      # U-Net depth-prediction pipeline
└── README.md
```

Both pipelines share the same camera model and dataset convention (FAT): pinhole
camera, no distortion, resolution 960 × 540, `fx = fy = 768.16`, `cx = 480`,
`cy = 270`, baseline `B = 0.06 m`, ground-truth depth stored as 16-bit PNG in
0.1 mm units (÷ 10000 → meters).

---

# Classical Approach (SGBM)

Classical stereo depth reconstruction using Semi-Global Block Matching, located in
`ClassicalApproach/sgbm/`.

## What it does

Given a calibrated stereo image pair, the pipeline:

1. **Rectifies** both images so epipolar lines are horizontal (`cv2.stereoRectify` + remap)
2. **Computes disparity** with OpenCV's `StereoSGBM` (left- and right-reference, then fuses them to fill the left-edge dead band)
3. **Converts disparity to depth** using `Z = fx × baseline / disparity`
4. **Back-projects** valid depth pixels into a 3-D point cloud (PLY)
5. **Evaluates** against ground-truth depth: AbsRel, RMSE, δ < 1.25, coverage

### Left-edge dead band fix

`StereoSGBM` can only search `numDisparities` columns to the right of each pixel,
so the leftmost 128 columns always come out empty. We fix this by running a second
SGBM pass with the images flipped horizontally (right-camera as reference), then
re-projecting those disparities back into left-image coordinates to fill the gap.
Coverage jumps from ~80% to ~95% with no accuracy loss on the originally valid
pixels.

## Folder structure

```
ClassicalApproach/
├── requirements.txt
└── sgbm/
    ├── config.py           # global paths and SGBM parameters (SgbmParams)
    ├── dense/
    │   ├── rectify.py      # stereo rectification (cv2.stereoRectify + remap)
    │   ├── sgbm.py         # SGBM matching, LR fusion, disparity → depth
    │   └── pipeline.py     # end-to-end dense stereo pipeline
    ├── eval/
    │   └── metrics.py      # AbsRel, RMSE, delta<1.25, coverage
    ├── io/
    │   ├── fat_io.py       # FAT camera loader, RGB + GT depth reader
    │   ├── depth_io.py     # save/load .npy depth maps
    │   └── ply_io.py       # write ASCII PLY point clouds
    ├── runners/
    │   ├── discover.py     # auto-discover dataset instances
    │   ├── run_one.py      # run pipeline for a single scene/frame
    │   ├── run_all.py      # run all instances, write metrics CSV
    │   ├── run_from_depth.py # evaluate a precomputed depth map
    │   └── visualize.py    # depth overlay, compare panel, 3-D point cloud
    └── tests/
        ├── test_fat_io.py  # smoke tests against live dataset
        └── test_metrics.py # unit tests for metric functions
```

## Dataset setup

Download the FAT dataset from the
[NVIDIA project page](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
and place scene folders under `ClassicalApproach/datasets/fat/`:

```
ClassicalApproach/datasets/fat/
├── Kitchen/
│   ├── _camera_settings.json
│   ├── 000000.left.jpg
│   ├── 000000.right.jpg
│   ├── 000000.left.depth.png
│   └── ...
├── Kitedemo/
└── Temple_1/
```

Each scene must contain `_camera_settings.json` plus matching `.left.jpg`,
`.right.jpg`, `.left.depth.png`, `.right.depth.png`, `.left.json`, and
`.right.json` files per frame.

## Installation

```bash
cd ClassicalApproach
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Usage

All commands are run from inside `ClassicalApproach/` (the `sgbm` package root):

**Run all scenes:**
```bash
cd ClassicalApproach
python -m sgbm.runners.run_all
```

**Run one scene/frame:**
```bash
python -m sgbm.runners.run_one Kitchen 000000
```

**Visualize results:**
```bash
# Depth overlay
python -m sgbm.runners.visualize depth Kitchen 000000

# 4-panel comparison (RGB | dense | GT | point cloud)
python -m sgbm.runners.visualize compare Kitchen 000000

# Full grid across all scenes
python -m sgbm.runners.visualize grid

# 3-D point cloud scatter + 2-D overlay
python -m sgbm.runners.visualize pointcloud Kitchen 000000

# Summary grid of all point clouds
python -m sgbm.runners.visualize pointcloud-grid
```

**Run tests:**
```bash
pytest sgbm/tests/
```

Outputs are written to `ClassicalApproach/outputs/sgbm_fat/<scene>/<frame>/`.

## Parameters

Key parameters are in `sgbm/config.py` → `SgbmParams`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `sgbm_num_disparities` | 128 | Disparity search range (px); also equals left dead-band width |
| `sgbm_block_size` | 5 | Matching block size; larger = smoother but loses fine edges |
| `sgbm_uniqueness_ratio` | 10 | Reject ambiguous matches (higher = stricter) |
| `depth_max_m` | 20.0 | Clip depth beyond this (meters) |

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

---

# Benchmark: Classical vs. Deep Learning

Best learned model vs. pretrained state-of-the-art monocular models on the FAT test set:

| Model | AbsRel ↓ | RMSE ↓ |
|---|---|---|
| **Residual_Unet_3D_5K (ours)** | **0.0725** | **8.10** |
| ZoeDepth-NK | 0.1622 | 16.99 |
| Depth-Anything-V2-Small | 0.7928 | 51.39 |
| DPT-Large | 0.8250 | 50.47 |

Head-to-head, the two paradigms are complementary: **SGBM** is dramatically more
accurate on near, textured surfaces and yields the cleanest metric geometry, while
the **deep model** is more robust on far, low-texture regions and reaches full
(100%) coverage by construction.
