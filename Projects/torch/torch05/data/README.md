# Dataset

The model was trained and evaluated using the **Intel Image Classification** dataset available on Kaggle.

- **Dataset Name:** Intel Image Classification
- **Source:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- **Total Images:** Approximately **25,000** RGB images
- **Image Size:** **150 × 150** pixels
- **Number of Classes:** **6**
  - Buildings
  - Forest
  - Glacier
  - Mountain
  - Sea
  - Street
- **Training Set:** Approximately **14,000** labelled images
- **Test Set:** Approximately **3,000** labelled images
- **Prediction Set:** Approximately **7,000** unlabelled images

## Dataset Description

The Intel Image Classification dataset consists of natural scene images collected from various locations around the world. It was originally released as part of an Intel image classification challenge and is widely used for benchmarking deep learning and transfer learning models for multi-class image classification tasks.

The images are organised into six cl  ass-specific folders, making the dataset suitable for supervised learning. Each image has a resolution of **150 × 150** pixels and is stored in RGB format. The dataset is compatible with popular deep learning frameworks such as TensorFlow, Keras, and PyTorch.

## Dataset Setup

1. Download the **Intel Image Classification** dataset from Kaggle:
   https://www.kaggle.com/datasets/puneet6060/intel-image-classification

2. Extract the downloaded archive.

3. Place the extracted dataset inside the project's `data/` directory so that the folder structure is as follows:

```text
project-root/
├── data/
│   └── intel-image-classification/
...
```

> **Note:** Ensure that the `seg_train`, `seg_test`, and `seg_pred` folders are **not** located directly inside the `data/` directory. The training scripts expect this directory structure by default.

## Class Labels

| Class | Description |
|--------|-------------|
| Buildings | Images containing residential or commercial buildings and urban structures |
| Forest | Images of forests, trees, and dense vegetation |
| Glacier | Images featuring glaciers, snow-covered ice formations, and frozen landscapes |
| Mountain | Images of mountains, hills, and rocky terrains |
| Sea | Images of oceans, beaches, and other water bodies |
| Street | Images of roads, highways, and urban streets |

## Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Images | ~25,000 |
| Image Resolution | 150 × 150 pixels |
| Colour Format | RGB |
| Number of Classes | 6 |
| Training Images | ~14,000 |
| Test Images | ~3,000 |
| Prediction Images | ~7,000 |

## Reference

Puneet Bansal. *Intel Image Classification Dataset*. Kaggle. Available at: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
