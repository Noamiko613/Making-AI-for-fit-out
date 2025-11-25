import os
import json
import logging
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")
PROCESSED_ROOT = Path("processed_data")

# Image settings
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def clean_directory(directory: Path):
    """
    Recursively removes non-image files and corrupt images.
    """
    logger.info(f"Cleaning directory: {directory}")
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    removed_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            
            # Remove non-image files
            if file_path.suffix.lower() not in valid_extensions:
                if file_path.name in ['styles.csv', 'images.csv', 'metadata.csv', 'class_dict.csv', 'list_attr_celeba.csv', 'list_bbox_celeba.csv', 'list_eval_partition.csv', 'list_landmarks_align_celeba.csv'] or file_path.suffix.lower() in ['.json', '.txt', '.csv']:
                    continue # Keep metadata files
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
                continue
            
            # Verify image integrity
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify it's an image
            except (IOError, SyntaxError, UnidentifiedImageError) as e:
                logger.warning(f"Corrupt image found: {file_path} - {e}")
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove corrupt file {file_path}: {e}")

    logger.info(f"Cleaned {directory}. Removed {removed_count} files.")

def process_fashion_dataset():
    """
    Process the paramaggarwal/fashion-product-images-small dataset.
    """
    dataset_dir = DATA_ROOT / "fashion-dataset"
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory {dataset_dir} not found. Skipping.")
        return

    images_dir = dataset_dir / "images"
    styles_path = dataset_dir / "styles.csv"

    if not styles_path.exists():
        # Try to find it recursively if structure is different
        found = list(dataset_dir.rglob("styles.csv"))
        if found:
            styles_path = found[0]
            images_dir = styles_path.parent / "images"
        else:
            logger.error("styles.csv not found in fashion-dataset.")
            return

    logger.info(f"Processing Fashion Dataset from {dataset_dir}")
    
    # Clean images directory
    clean_directory(images_dir)

    # Load metadata
    try:
        df = pd.read_csv(styles_path, on_bad_lines='skip')
    except Exception as e:
        logger.error(f"Failed to read styles.csv: {e}")
        return

    # Filter for apparel only (optional, based on user needs)
    # For now, we keep everything but ensure image exists
    
    valid_data = []
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Normalization will happen in the Dataset class during training, 
        # but we can pre-process if we want to save tensors (takes more space).
        # Here we just verify we can open and resize.
    ])

    logger.info("Verifying and resizing images...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_id = str(row['id'])
        img_path = images_dir / f"{img_id}.jpg"
        
        if not img_path.exists():
            continue
            
        try:
            # We don't save the resized image to disk to save space/time now, 
            # but we could. For now, just verifying it's loadable.
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                # img = transform(img) # Just testing transform
            
            valid_data.append(row)
        except Exception:
            continue

    valid_df = pd.DataFrame(valid_data)
    logger.info(f"Valid images: {len(valid_df)} / {len(df)}")
    
    # Save processed metadata
    PROCESSED_ROOT.mkdir(exist_ok=True)
    valid_df.to_csv(PROCESSED_ROOT / "fashion_processed.csv", index=False)
    logger.info(f"Saved processed metadata to {PROCESSED_ROOT / 'fashion_processed.csv'}")

if __name__ == "__main__":
    process_fashion_dataset()
def process_polyvore_dataset():
    """
    Process the Polyvore Outfits dataset.
    """
    dataset_dir = DATA_ROOT / "polyvore-outfits"
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory {dataset_dir} not found. Skipping.")
        return

    logger.info(f"Processing Polyvore Dataset from {dataset_dir}")
    # Polyvore structure is complex (images inside subfolders, JSON metadata).
    # For now, we just clean the images directory.
    
    # Find images directory (it might be nested)
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        # Try to find 'images' folder recursively
        found = list(dataset_dir.rglob("images"))
        if found:
            images_dir = found[0]
    
    if images_dir and images_dir.exists():
        clean_directory(images_dir)
    else:
        logger.warning("Images directory not found in Polyvore dataset.")

def process_celeba_dataset():
    """
    Process the CelebA dataset.
    """
    dataset_dir = DATA_ROOT / "celeba-dataset"
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory {dataset_dir} not found. Skipping.")
        return

    logger.info(f"Processing CelebA Dataset from {dataset_dir}")
    
    images_dir = dataset_dir / "img_align_celeba" / "img_align_celeba"
    if not images_dir.exists():
         # Try one level up or down depending on extraction
         images_dir = dataset_dir / "img_align_celeba"
    
    if images_dir.exists():
        clean_directory(images_dir)
    else:
        logger.warning("Images directory not found in CelebA dataset.")

def process_segmentation_dataset():
    """
    Process the Clothing Co-Parsing dataset.
    """
    dataset_dir = DATA_ROOT / "clothing-segmentation"
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory {dataset_dir} not found. Skipping.")
        return

    logger.info(f"Processing Segmentation Dataset from {dataset_dir}")
    
    # Clean recursively as structure varies
    clean_directory(dataset_dir)

if __name__ == "__main__":
    process_fashion_dataset()
    process_polyvore_dataset()
    process_celeba_dataset()
    process_segmentation_dataset()
