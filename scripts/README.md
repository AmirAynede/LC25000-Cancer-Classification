# Scripts Overview — LC25000 Cancer Classification Project

This folder contains all modular Python scripts used in the LC25000 Cancer Classification pipeline.
Each script represents a self-contained stage of the end-to-end deep learning workflow for histopathology image classification using the LC25000 dataset.
All scripts can be run individually or imported as modules into notebooks.

## 1. Core Utilities

| Script           | Description                                                                                        | Example Usage                         |
| :--------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------ |
| **`dataset.py`** | Defines data transformations and PyTorch `DataLoader` setup for training, validation, and testing. | `from dataset import get_dataloaders` |
| **`model.py`**   | Builds the CNN model (ResNet18, ResNet34, EfficientNet-B0) for transfer learning.                  | `from model import build_model`       |

## 2. Dataset Preparation

| Script                 | Description                                                                                                  | Example Usage                     |
| :--------------------- | :----------------------------------------------------------------------------------------------------------- | :-------------------------------- |
| **`split_dataset.py`** | Splits the raw LC25000 dataset into train, validation, and test directories while maintaining class balance. | `python -m scripts.split_dataset` |


## 3. Model Exploration & Training

| Script                 | Description                                                                                             | Example Usage                     |
| :--------------------- | :------------------------------------------------------------------------------------------------------ | :-------------------------------- |
| **`model_summary.py`** | Prints a detailed layer-by-layer summary of the chosen model using `torchinfo`.                         | `python -m scripts.model_summary` |
| **`train.py`**         | Trains the model with early stopping and learning rate scheduling. Saves best weights and metrics JSON. | `python -m scripts.train`         |


## 4. Metrics & Visualization

| Script                           | Description                                                                                                 | Example Usage                               |
| :------------------------------- | :---------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| **`plot.py`**                    | Plots training and validation accuracy/loss curves from JSON logs.                                          | `python -m scripts.plot`                    |
| **`animate_training_curves.py`** | Generates an animated visualization of training progression.                                                | `python -m scripts.animate_training_curves` |
| **`evaluate_on_test.py`**        | Evaluates the model on the test set and saves classification report, confusion matrix, and predictions CSV. | `python -m scripts.evaluate_on_test`        |


## 5. Visual Analysis

| Script                                | Description                                                            | Example Usage                                    |
| :------------------------------------ | :--------------------------------------------------------------------- | :----------------------------------------------- |
| **`visualize_predictions.py`**        | Displays a grid of random test predictions with true/predicted labels. | `python -m scripts.visualize_predictions`        |
| **`visualize_misclassifications.py`** | Shows only misclassified images for qualitative error analysis.        | `python -m scripts.visualize_misclassifications` |


## 6.Explainability (Grad-CAM)

| Script                          | Description                                                                                        | Example Usage                              |
| :------------------------------ | :------------------------------------------------------------------------------------------------- | :----------------------------------------- |
| **`gradcam.py`**                | Generates Grad-CAM heatmaps highlighting diagnostic regions influencing predictions.               | `python -m scripts.gradcam`                |
| **`visualize_gradcam_grid.py`** | Compares original and Grad-CAM images side-by-side for several samples (e.g., misclassifications). | `python -m scripts.visualize_gradcam_grid` |


## Notes

All paths assume project structure:

```bash
LC25000-Cancer-Classification/
├── data/
├── results/
├── outputs/
├── saved_models/
└── scripts/
```

Run scripts from the project root using:

`python -m scripts.<script_name>`

Each script auto-creates required folders `(outputs/, results/, saved_models/)` if missing.






