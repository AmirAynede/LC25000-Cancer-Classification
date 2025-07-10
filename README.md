# LC25000 Cancer Classification - Full ML Pipeline

[![Notebook](https://img.shields.io/badge/notebook-ipynb-green?style=flat-square)](https://github.com/AmirAynede/LC25000-Cancer-Classification/tree/main/notebook)
[![Scripts](https://img.shields.io/badge/scripts-python-blue?style=flat-square)](https://github.com/AmirAynede/LC25000-Cancer-Classification/tree/main/scripts)
[![Report](https://img.shields.io/badge/report-pdf-red?style=flat-square)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/LC25000_Cancer_Classification_Report.pdf)
[![Citation](https://img.shields.io/badge/citation-CFF-white?style=flat-square)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/CITATION.cff)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange?style=flat-square)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/LICENSE.md)
[![Requirements](https://img.shields.io/badge/requirements-necessary-darkgreen?style=flat-square)](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/requirements.txt)

![Project Cover](https://github.com/AmirAynede/LC25000-Cancer-Classification/blob/main/COVER.PNG?raw=true)

This repository provides a complete, reproducible pipeline for training, evaluating, and interpreting a deep learning model on the LC25000 histopathology dataset.

---

## Table of Contents

0. Project Directory Setup
1. Environment Setup
2. Data Download & Extraction
   - 2.1 Set Working Directory
   - 2.2 Place kaggle.json
   - 2.3 Download and Extract Dataset
3. Dataset Splitting
4. Model Summary
5. Training
6. Plotting Training Metrics
7. Animated Training Curves
8. Evaluation on Test Set
9. Visualize Predictions as an Image Grid
10. Visualize Misclassifications
11. Grad-CAM Visualization
12. Grad-CAM Grid (side-by-side)

---

## 0. Project Directory Setup
**Purpose:** Ensure all required folders exist so scripts and outputs work without errors.

**Process:**
- Run the setup cell in the notebook or execute the following in Python:
  ```python
  import os
  folders = ['data', 'notebooks', 'outputs', 'results', 'sample_images', 'saved_models', 'scripts']
  for folder in folders:
      os.makedirs(folder, exist_ok=True)
  ```

**Result:**
- Folders for data, outputs, results, models, scripts, etc. are created.

---

## 1. Environment Setup
**Purpose:** Install all required Python dependencies.

**Process:**
- Run:
  ```bash
  pip install -r requirements.txt
  ```
- For M1/M2 Mac GPU support, run:
  ```bash
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  ```

**Result:**
- All necessary packages are installed for the pipeline to run.

---

## 2. Data Download & Extraction (Kaggle)

### 2.1 Set Working Directory
**Purpose:** Ensure your notebook or script is running from the project root so all file paths work correctly.

**Instructions:**
- In your notebook, run:
  ```python
  import os
  os.chdir('/.../.../cancer_clasification_lc25000')
  print("Current working directory:", os.getcwd())
  ```
- Replace `/.../.../cancer_clasification_lc25000` with the actual path to your project root if needed.

---

### 2.2 Place kaggle.json
**Purpose:** Provide your Kaggle API credentials for dataset download.

**Instructions:**
- Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account) and click "Create New API Token" to download `kaggle.json`.
- Place `kaggle.json` in your project root directory (the same directory as your notebook or script).

---

### 2.3 Download and Extract Dataset
**Purpose:** Download the LC25000 dataset from Kaggle and extract it for use in the pipeline.

**Instructions:**
- Install the Kaggle CLI:
  ```bash
  pip install kaggle
  ```
- Move `kaggle.json` to the correct location and set permissions:
  ```bash
  mkdir -p ~/.kaggle
  mv kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  ```
- Download and unzip the dataset:
  ```bash
  mkdir -p data
  kaggle datasets download andrewmvd/lung-and-colon-cancer-histopathological-images -p data/
  unzip -q data/lung-and-colon-cancer-histopathological-images.zip -d data/Lung_and_Colon_Cancer
  ```
- **Skip if you already have `data/Lung_and_Colon_Cancer/`.**

**Result:**
- Raw images are available in `data/Lung_and_Colon_Cancer/`.

**Troubleshooting:**
- If you get a `FileNotFoundError` for `kaggle.json`, make sure your working directory is set to your project root and that `kaggle.json` is present there before running the commands above.

---

## 3. Dataset Splitting
**Purpose:** Split the raw dataset into train/val/test sets for reproducible experiments.

**Process:**
- Run:
  ```bash
  python -m scripts.split_dataset
  ```
- This creates `data/lc25000_split/` with `train/`, `val/`, and `test/` subfolders.

**Result:**
- Data is organized for training, validation, and testing.

---

## 4. Model Summary
**Purpose:** Review the architecture, output shapes, and parameter counts of the model.

**Process:**
- Run:
  ```bash
  python -m scripts.model_summary --num_classes 5 --input_size 1 3 224 224
  ```
- Adjust arguments if your data/model shape is different.

**Result:**
- A detailed summary table of the model is printed.

---

## 5. Training
**Purpose:** Train the ResNet18 model on the LC25000 dataset.

**Process:**
- Run:
  ```bash
  python -m scripts.train
  ```
- The script will save the best model to `saved_models/` and training metrics to `results/`.

**Result:**
- Trained model weights and training metrics are saved for later use.

---

## 6. Plotting Training Metrics
**Purpose:** Visualize loss and accuracy curves to monitor training progress.

**Process:**
- Run:
  ```bash
  python -m scripts.plot --json_path results/training_metrics_<timestamp>.json
  ```
- Replace `<timestamp>` with your actual metrics file.

**Result:**
- Plots are saved in `outputs/` for loss and accuracy.

---

## 7. Animated Training Curves
**Purpose:** See an animated visualization of how loss and accuracy evolve over epochs.

**Process:**
- Run:
  ```bash
  python -m scripts.animate_training_curves
  ```
- The latest metrics file is used automatically.

- This step animates the training and validation loss and accuracy curves over epochs, so you can visually see how your model improves during training.
How to use:
You can either:

Run the Python script directly (in a terminal):
python -m scripts.animate_training_curves
This will display the animation in a separate window (best for local use).

OR

Copy and run the provided code cell in your notebook to see the animation inline in the notebook output (recommended for Jupyter/Colab).

Tip:

The notebook cell version is best for interactive exploration.
The script version is useful for automated runs or when working outside a notebook.

**Result:**
- An animation of the training curves is displayed.

---

## 8. Evaluation on Test Set
**Purpose:** Evaluate the trained model on the test set and save detailed results.

**Process:**
- Run:
  ```bash
  python -m scripts.evaluate_on_test
  ```
- Outputs:
  - `outputs/classification_report.txt`: Precision, recall, F1-score per class
  - `outputs/confusion_matrix.png`: Confusion matrix plot
  - `outputs/test_predictions.csv`: Per-image predictions (filename, true label, predicted label)

**Result:**
- Quantitative evaluation and per-image predictions for further analysis.

---

## 9. Visualize Predictions as an Image Grid
**Purpose:** Visually inspect a random sample of test predictions.

**Process:**
- Run:
  ```bash
  python -m scripts.visualize_predictions --csv_path outputs/test_predictions.csv --n_images 9 --cols 3 --output_path outputs/prediction_grid.png
  ```
- Adjust `--n_images` and `--cols` as desired.

**Result:**
- A grid of test images with true and predicted labels is displayed and saved.

---

## 10. Visualize Misclassifications
**Purpose:** Focus on and analyze the images the model got wrong.

**Process:**
- Run:
  ```bash
  python -m scripts.visualize_misclassifications --csv_path outputs/test_predictions.csv --n_images 9 --cols 3 --output_path outputs/misclassified_grid.png
  ```

**Result:**
- A grid of misclassified images is displayed and saved for error analysis.

---

## 11. Grad-CAM Visualization
**Purpose:** Interpret model predictions by visualizing which parts of the image influenced the decision.

**Process:**
- Run:
  ```bash
  python -m scripts.gradcam --image_path <path_to_image> --model_path <path_to_model>
  ```
- Replace `<path_to_image>` and `<path_to_model>` as needed.

**Result:**
- Grad-CAM heatmap is saved in `outputs/` for the selected image.

---

## 12. Grad-CAM Grid (side-by-side)
**Purpose:** Compare original images and Grad-CAM heatmaps for a set of (optionally misclassified) images.

**Process:**
- Run:
  ```bash
  python -m scripts.visualize_gradcam_grid --csv_path outputs/test_predictions.csv --model_path <path_to_model> --n_images 4 --cols 2 --only_misclassified --output_path outputs/gradcam_grid.png
  ```
- Adjust arguments as needed.

**Result:**
- A grid of original and Grad-CAM images is displayed and saved for qualitative analysis.

---

## Reproducibility & Tips
- Always run the steps in order for a clean workflow.
- If you change the dataset or scripts, re-run the relevant steps.
- All outputs are saved in the appropriate folders for easy access and sharing.

---
## Run the Project on Google Colab

If you prefer running the LC25000 cancer classification workflow step by step in a cloud environment (no local setup required), use the dedicated Google Colab notebook below:

▶[Open the LC25000 Classification Colab Notebook](https://colab.research.google.com/drive/1J3jZgfGz3SBH9HkTRtAZb2LEthYnC_Os)

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
