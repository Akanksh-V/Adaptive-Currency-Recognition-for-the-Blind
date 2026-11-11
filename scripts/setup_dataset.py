#!/usr/bin/env python3
"""
Dataset Setup Script for Adaptive Currency Recognition (Blind Assist)
This script helps download and extract the Indian Currency dataset from Kaggle
into the expected dataset/ folder structure.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DOWNLOADS_DIR = Path.home() / "Downloads"

KAGGLE_DATASET_NAME = "vishalmane109/indian-currency-note-images-dataset-2020"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_NAME}"

CLASS_NAMES = [
    "1Hundrednote",
    "2Hundrednote",
    "2Thousandnote",
    "5Hundrednote",
    "Fiftynote",
    "Tennote",
    "Twentynote"
]

def ensure_dirs():
    """Ensure target folders exist."""
    for split in ["training", "validation", "test"]:
        for cls in CLASS_NAMES:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"Verified dataset directory structure at {DATASET_DIR}")

def find_local_zip():
    """Look for an existing downloaded zip in Downloads or current folder."""
    candidates = list(DOWNLOADS_DIR.glob("*currency*.zip")) + \
                 list(DOWNLOADS_DIR.glob("*indian*.zip")) + \
                 list(BASE_DIR.glob("*.zip"))
    return candidates[0] if candidates else None

def extract_and_organize(zip_path):
    """Extract and move images to dataset/."""
    print(f"Extracting {zip_path}...")
    temp_extract = BASE_DIR / "temp_dataset_extract"
    temp_extract.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_extract)
        
    print("Organizing folders...")
    # Search for training/validation/test folders inside extraction
    for root, dirs, files in os.walk(temp_extract):
        for d in dirs:
            low = d.lower()
            if low in ["training", "train"]:
                target_split = DATASET_DIR / "training"
                src = Path(root) / d
                for c in src.iterdir():
                    if c.is_dir():
                        dest_c = target_split / c.name
                        dest_c.mkdir(parents=True, exist_ok=True)
                        for img in c.glob("*.*"):
                            shutil.copy2(img, dest_c / img.name)
            elif low in ["validation", "val"]:
                target_split = DATASET_DIR / "validation"
                src = Path(root) / d
                for c in src.iterdir():
                    if c.is_dir():
                        dest_c = target_split / c.name
                        dest_c.mkdir(parents=True, exist_ok=True)
                        for img in c.glob("*.*"):
                            shutil.copy2(img, dest_c / img.name)
            elif low in ["test", "testing"]:
                target_split = DATASET_DIR / "test"
                src = Path(root) / d
                for img in src.glob("*.*"):
                    shutil.copy2(img, target_split / img.name)

    shutil.rmtree(temp_extract, ignore_errors=True)
    print("Dataset extracted and organized successfully!")

def main():
    ensure_dirs()
    zip_file = find_local_zip()
    if zip_file:
        print(f"Found dataset archive at: {zip_file}")
        extract_and_organize(zip_file)
    else:
        print("\n" + "="*70)
        print("NO DATASET ARCHIVE FOUND LOCALLY.")
        print("The raw dataset is hosted on Kaggle (~4,000 images):")
        print(f"  URL: {KAGGLE_URL}")
        print("\nOption 1 (Direct Browser Download):")
        print(f"  1. Open: {KAGGLE_URL}")
        print("  2. Click 'Download' (archive size is approx. 500MB - 1GB)")
        print(f"  3. Save the .zip file to: {DOWNLOADS_DIR}")
        print("  4. Re-run this script: python3 scripts/setup_dataset.py")
        print("\nOption 2 (Kaggle CLI - if configured):")
        print(f"  kaggle datasets download -d {KAGGLE_DATASET_NAME} -p ~/Downloads")
        print("  python3 scripts/setup_dataset.py")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
