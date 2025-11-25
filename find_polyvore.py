import os
import logging
from pathlib import Path

# Set KAGGLE_CONFIG_DIR
local_kaggle_dir = Path.cwd() / ".kaggle"
if local_kaggle_dir.exists() and (local_kaggle_dir / "kaggle.json").exists():
    os.environ['KAGGLE_CONFIG_DIR'] = str(local_kaggle_dir)

import kaggle

def search_polyvore():
    print("Searching for 'polyvore' datasets...")
    datasets = kaggle.api.dataset_list(search="polyvore outfits")
    for d in datasets:
        print(f"Ref: {d.ref}, Title: {d.title}")

if __name__ == "__main__":
    search_polyvore()
