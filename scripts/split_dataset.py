import os
import shutil
import random
from pathlib import Path

# Original dataset folder
SOURCE_DIR = Path("data/Lung_and_Colon_Cancer")
TARGET_DIR = Path("data/lc25000_split")
SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = [0.7, 0.15, 0.15]  # 70/15/15 split

def create_split_dirs(classes):
    for split in SPLITS:
        for class_name in classes:
            split_dir = TARGET_DIR / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

def split_data():
    class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    classes = [d.name for d in class_dirs]

    create_split_dirs(classes)

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpeg"))
        random.shuffle(images)

        train_end = int(SPLIT_RATIOS[0] * len(images))
        val_end = train_end + int(SPLIT_RATIOS[1] * len(images))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, image_paths in splits.items():
            for img_path in image_paths:
                target_path = TARGET_DIR / split / class_dir.name / img_path.name
                shutil.copy(img_path, target_path)

        print(f"✅ Split class '{class_dir.name}': {len(images)} images")

if __name__ == "__main__":
    random.seed(42)
    split_data()
    print("✅ Dataset split completed!")