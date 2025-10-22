# LC25000 Cancer Classification - Full ML Pipeline

[![Notebook](https://img.shields.io/badge/Notebook-ipynb-green?style=flat-square&logo=jupyter&logoColor=white)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/notebook/ML_Pipeline.ipynb)
[![Scripts](https://img.shields.io/badge/Scripts-Python-blue?style=flat-square&logo=python&logoColor=white)](https://github.com/AmirAynede/LC25000-Cancer-Classification/tree/main/scripts)
[![Report](https://img.shields.io/badge/Report-PDF-red?style=flat-square&logo=readthedocs&logoColor=white)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/LC25000_Cancer_Classification%20.pdf)
[![Citation](https://img.shields.io/badge/Citation-CFF-lightgrey?style=flat-square&logo=academia&logoColor=black)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/CITATION.cff)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-orange?style=flat-square&logo=creativecommons&logoColor=white)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/LICENSE.md)
[![Requirements](https://img.shields.io/badge/Requirements-Necessary-darkgreen?style=flat-square&logo=dependabot&logoColor=white)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/requirements.txt)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Run_Notebook-yellow?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/17j1wBcvBr9qrTg_jWuEQdglIxgAaBuBc?usp=sharing)

![Project Cover](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/COVER.PNG?raw=true)

This repository provides a complete, reproducible pipeline for training, evaluating, and interpreting a deep learning model on the LC25000 histopathology dataset.

---
# An End-to-End Deep Learning Pipeline for Automated Histopathology Image Classification

## Overview
This project presents a complete and reproducible deep learning pipeline for automated cancer subtype classification using the LC25000 histopathology dataset.

It integrates all essential stages—from data preprocessing and model training to evaluation, visualization, and explainability—into a modular, transparent, and scalable framework built with PyTorch.

The pipeline classifies five histopathological tissue classes:

Colon Adenocarcinoma `(colon_aca)`

Benign Colon Tissue `(colon_n)`

Lung Adenocarcinoma `(lung_aca)`

Lung Squamous Cell `Carcinoma (lung_scc)`

Benign Lung Tissue `(lung_n)`


## Abstract
Manual histopathology examination remains the gold standard in cancer diagnosis but is time-consuming and subjective.
This project leverages transfer learning with a fine-tuned ResNet18 model to automate and standardize diagnostic workflows.
Interpretability is achieved via Grad-CAM visualizations, highlighting diagnostically relevant regions.
The pipeline achieved near-perfect accuracy (≈1.00) across all classes, demonstrating strong generalization, transparency, and clinical relevance

## Pipeline Structure
| Stage | Description                                |
| ----- | ------------------------------------------ |
| 1     | Project Directory Setup                    |
| 2     | Environment Setup                          |
| 3     | Dataset Download & Extraction (via Kaggle) |
| 4     | Dataset Splitting (train / val / test)     |
| 5     | Model Summary                              |
| 6     | Training                                   |
| 7     | Plot Training Metrics                      |
| 8     | Animated Training Curves                   |
| 9     | Evaluation on Test Set                     |
| 10     | Visualize Predictions Grid                |
| 11    | Visualize Misclassifications               |
| 12    | Grad-CAM Visualization                     |
| 13    | Grad-CAM Grid Comparison                   |

Each step is executable as a standalone script or interactively via the provided Jupyter notebook `(ML_Pipeline.ipynb)`.

## 1-3. Environment & Data Setup
Directory Initialization

```python
import os
folders = ['data', 'notebooks', 'outputs', 'results', 'sample_images', 'saved_models', 'scripts']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
```
Environment Installation

```
pip install -r requirements.txt
```

For M1/M2 Mac GPU support

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

Dataset Download (Kaggle)

```
pip install kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download andrewmvd/lung-and-colon-cancer-histopathological-images -p data/
unzip -q data/lung-and-colon-cancer-histopathological-images.zip -d data/Lung_and_Colon_Cancer
```

## 4–9. Training & Evaluation

Dataset Splitting

```
python -m scripts.split_dataset
```

Model Summary

```
python -m scripts.model_summary --num_classes 5 --input_size 1 3 224 224
```

Training

```
python -m scripts.train
```

Note: Saves best model weights in saved_models/ and metrics in results/.

Plot & Animate Training

```
python -m scripts.plot
python -m scripts.animate_training_curves
```

Evaluation on Test Set

```
python -m scripts.evaluate_on_test
```

Outputs (saved automatically):

| File                        | Description           | Location   |
| --------------------------- | --------------------- | ---------- |
| `classification_report.txt` | Precision, Recall, F1 | `results/` |
| `confusion_matrix.png`      | Confusion matrix      | `outputs/` |
| `test_predictions.csv`      | Per-image predictions | `results/` |

## 10–13. Visualization & Explainability

Predictions Grid

```
python -m scripts.visualize_predictions --csv_path results/test_predictions.csv --n_images 9 --cols 3 --output_path outputs/prediction_grid.png
```

Misclassifications Grid

```
python -m scripts.visualize_misclassifications --csv_path results/test_predictions.csv --n_images 9 --cols 3 --output_path outputs/misclassified_grid.png
```

Grad-CAM (Single Image)

```
python -m scripts.gradcam --image_path <path_to_image> --model_path <path_to_model>
```

Grad-CAM Grid

```
python -m scripts.visualize_gradcam_grid --csv_path results/test_predictions.csv --model_path <path_to_model> --n_images 4 --cols 2 --only_misclassified --output_path outputs/gradcam_grid.png
```

## Directory Structure

LC25000-Cancer-Classification/

```bash
│
├── data/                     # Raw + split datasets
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Modular Python scripts
├── results/                  # Metrics, CSVs, and reports
├── outputs/                  # Visual artifacts (plots, Grad-CAMs, confusion matrices)
├── saved_models/             # Trained model weights
└── requirements.txt
```

## Results Summary

| Class                        | Precision |   Recall  |  F1-score | Support |
| :--------------------------- | :-------: | :-------: | :-------: | :-----: |
| Colon Adenocarcinoma         |    1.00   |    1.00   |    1.00   |    73   |
| Benign Colon Tissue          |    1.00   |    1.00   |    1.00   |    85   |
| Lung Adenocarcinoma          |    1.00   |    0.98   |    0.99   |    63   |
| Benign Lung Tissue           |    1.00   |    1.00   |    1.00   |    76   |
| Lung Squamous Cell Carcinoma |    0.99   |    1.00   |    0.99   |    78   |
| **Overall Accuracy**         | **≈1.00** | **≈1.00** | **≈1.00** |   375   |

Model: ResNet18 (fine-tuned)

Optimizer: AdamW, LR = 1e-3

Hardware: NVIDIA T4 GPU (Google Colab)

Training Epochs: 30 (early stopping around epoch 26)


## Key Features:

Reproducible pipeline — modular scripts, deterministic splits, and CI support

Explainable AI — Grad-CAM overlays highlight clinically meaningful regions

EDA-driven design — brightness, class balance, and augmentation verified

Comparative models — ResNet18, ResNet34, EfficientNet-B0

High interpretability — model attention aligns with diagnostic tissue structures

## Limitations & Future Work
LC25000 consists of cropped patches, not full whole-slide images (WSIs).

Future extensions include:

Whole-slide image aggregation (WSI-level classification)

Domain adaptation and stain normalization

Multi-institutional data testing

Uncertainty estimation and model calibration

Deployment via Flask/FastAPI with real-time Grad-CAM support



## Run the Project on Google Colab

If you prefer running the LC25000 cancer classification workflow step by step in a cloud environment (no local setup required), use the dedicated Google Colab notebook below:

▶[Open the LC25000 Classification Colab Notebook](https://colab.research.google.com/drive/17j1wBcvBr9qrTg_jWuEQdglIxgAaBuBc?usp=sharing)

Features:

   No installation needed
   
   GPU support available on Colab
   
   All steps: dataset download, preprocessing, model training, evaluation, Grad-CAM

How to Use:

   Click the link to open the notebook in Google Colab.
   
   Follow each cell in order, from environment setup to final visualization.
   
   Upload your kaggle.json when prompted to enable dataset download from Kaggle.
   
   Run all cells to reproduce the results and visualizations.
   

   Make sure you are signed in to your Google account to use Colab, and enable GPU under Runtime > Change runtime type > Hardware Accelerator.

---

## Citation
If you use this pipeline, please cite the original LC25000 dataset and this repository.

---

## License
This project is for academic and research use. Dataset usage must comply with original terms of use. 
