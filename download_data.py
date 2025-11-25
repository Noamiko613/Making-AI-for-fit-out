import os
import logging
from pathlib import Path

# Set KAGGLE_CONFIG_DIR to the local .kaggle directory if it exists there
# This must be done BEFORE importing kaggle
local_kaggle_dir = Path.cwd() / ".kaggle"
if local_kaggle_dir.exists() and (local_kaggle_dir / "kaggle.json").exists():
    os.environ['KAGGLE_CONFIG_DIR'] = str(local_kaggle_dir)

import kaggle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define datasets to download
# Format: (Kaggle Dataset ID, Destination Folder Name)
DATASETS = [
    ("paramaggarwal/fashion-product-images-small", "fashion-dataset"),
    ("dnepozitek/polyvore-outfits", "polyvore-outfits"),
    ("jessicali9530/celeba-dataset", "celeba-dataset"),
    ("balraj98/clothing-coparsing-dataset", "clothing-segmentation")
]

DATA_ROOT = Path("data")

def download_datasets():
    """
    Downloads the required datasets from Kaggle using the official API.
    """
    # Ensure data directory exists
    if not DATA_ROOT.exists():
        DATA_ROOT.mkdir(parents=True)
        logger.info(f"Created data directory at {DATA_ROOT.absolute()}")

    # Set KAGGLE_CONFIG_DIR to the local .kaggle directory if it exists there
    local_kaggle_dir = Path.cwd() / ".kaggle"
    if local_kaggle_dir.exists() and (local_kaggle_dir / "kaggle.json").exists():
        os.environ['KAGGLE_CONFIG_DIR'] = str(local_kaggle_dir)
        logger.info(f"Using local Kaggle config from {local_kaggle_dir}")

    # Authenticate (relies on ~/.kaggle/kaggle.json or environment variables)
    try:
        kaggle.api.authenticate()
        logger.info("Kaggle API authenticated successfully.")
    except Exception as e:
        logger.error(f"Failed to authenticate with Kaggle API: {e}")
        logger.error("Please ensure you have placed 'kaggle.json' in C:\\Users\\<User>\\.kaggle\\ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return

    for dataset_id, folder_name in DATASETS:
        target_dir = DATA_ROOT / folder_name
        
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Dataset '{dataset_id}' appears to be already downloaded in {target_dir}. Skipping.")
            continue

        logger.info(f"Downloading '{dataset_id}' to {target_dir}...")
        try:
            kaggle.api.dataset_download_files(dataset_id, path=target_dir, unzip=True)
            logger.info(f"Successfully downloaded and extracted '{dataset_id}'.")
        except Exception as e:
            logger.error(f"Failed to download '{dataset_id}': {e}")

if __name__ == "__main__":
    logger.info("Starting data acquisition process...")
    download_datasets()
    logger.info("Data acquisition completed.")
