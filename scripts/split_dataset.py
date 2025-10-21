# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: split_dataset.py
# Description: Split LC25000 dataset into train/val/test folders

import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cls in os.listdir(input_dir):
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(split_dir, f))

if __name__ == "__main__":
    input_dir = "data/Lung_and_Colon_Cancer"
    output_dir = "data/lc25000_split"
    split_dataset(input_dir, output_dir)
