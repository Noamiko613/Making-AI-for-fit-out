import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")
PROCESSED_ROOT = Path("processed_data")
LABEL_MAP_PATH = Path("label_map.json")

def load_label_map():
    with open(LABEL_MAP_PATH, 'r') as f:
        return json.load(f)

def process_fashion_dataset(label_map):
    logger.info("Unifying Fashion Dataset...")
    input_path = PROCESSED_ROOT / "fashion_processed.csv"
    if not input_path.exists():
        logger.warning(f"{input_path} not found. Skipping Fashion Dataset.")
        return

    df = pd.read_csv(input_path)
    
    unified_data = []
    mapping = label_map['mappings']['fashion_dataset']['articleType']
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Fashion"):
        original_label = row['articleType']
        if original_label in mapping:
            unified_category = mapping[original_label]
            unified_data.append({
                'image_path': str(DATA_ROOT / "fashion-dataset" / "images" / f"{row['id']}.jpg"),
                'category': unified_category,
                'original_category': original_label,
                'color': row['baseColour'],
                'season': row['season'],
                'usage': row['usage'],
                'source': 'fashion_dataset'
            })
            
    df_unified = pd.DataFrame(unified_data)
    
    # Split
    train, test = train_test_split(df_unified, test_size=0.2, stratify=df_unified['category'], random_state=42)
    val, test = train_test_split(test, test_size=0.5, stratify=test['category'], random_state=42)
    
    train.to_csv(PROCESSED_ROOT / "train_fashion.csv", index=False)
    val.to_csv(PROCESSED_ROOT / "val_fashion.csv", index=False)
    test.to_csv(PROCESSED_ROOT / "test_fashion.csv", index=False)
    logger.info(f"Saved Fashion splits. Train: {len(train)}")

def process_polyvore_dataset(label_map):
    logger.info("Unifying Polyvore Dataset...")
    polyvore_root = DATA_ROOT / "polyvore-outfits" / "polyvore_outfits"
    meta_path = polyvore_root / "polyvore_item_metadata.json"
    train_json_path = polyvore_root / "disjoint" / "train.json"
    
    if not meta_path.exists() or not train_json_path.exists():
        logger.warning("Polyvore metadata or train.json not found. Skipping.")
        return

    # Load item metadata
    with open(meta_path, 'r') as f:
        item_meta = json.load(f)
        
    # Load outfits
    with open(train_json_path, 'r') as f:
        outfits = json.load(f)
        
    processed_outfits = []
    
    for outfit in tqdm(outfits, desc="Processing Polyvore Outfits"):
        set_id = outfit['set_id']
        items = []
        for item in outfit['items']:
            item_id = item['item_id']
            # Check if image exists
            img_path = polyvore_root / "images" / f"{item_id}.jpg"
            if not img_path.exists():
                continue
                
            # Get category
            meta = item_meta.get(item_id, {})
            semantic_category = meta.get('semantic_category', 'unknown')
            
            items.append({
                'item_id': item_id,
                'image_path': str(img_path),
                'category': semantic_category,
                'description': meta.get('description', '') # Useful for LLM/Text
            })
            
        if len(items) > 1: # Only keep outfits with at least 2 items
            processed_outfits.append({
                'set_id': set_id,
                'items': items
            })
            
    # Save as JSON (nested structure better for compatibility)
    with open(PROCESSED_ROOT / "train_polyvore.json", 'w') as f:
        json.dump(processed_outfits, f, indent=2)
        
    logger.info(f"Saved {len(processed_outfits)} Polyvore outfits.")

def process_celeba_dataset():
    logger.info("Unifying CelebA Dataset...")
    celeba_root = DATA_ROOT / "celeba-dataset"
    attr_path = celeba_root / "list_attr_celeba.csv"
    images_dir = celeba_root / "img_align_celeba" / "img_align_celeba"
    
    if not attr_path.exists():
        logger.warning("CelebA attributes file not found. Skipping.")
        return
        
    df = pd.read_csv(attr_path)
    
    # Filter relevant attributes for appearance
    # 1 means present, -1 means absent
    relevant_cols = ['image_id', 'Male', 'Young', 'Pale_Skin', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald', 'Oval_Face', 'Chubby']
    df = df[relevant_cols]
    
    # Add image path
    df['image_path'] = df['image_id'].apply(lambda x: str(images_dir / x))
    
    # Verify images exist (check first 100 to be fast, or all if rigorous)
    # For speed, we assume if folder exists, most are there.
    if not images_dir.exists():
         # Try one level up
         images_dir = celeba_root / "img_align_celeba"
         df['image_path'] = df['image_id'].apply(lambda x: str(images_dir / x))
    
    # Split
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    
    train.to_csv(PROCESSED_ROOT / "train_celeba.csv", index=False)
    val.to_csv(PROCESSED_ROOT / "val_celeba.csv", index=False)
    test.to_csv(PROCESSED_ROOT / "test_celeba.csv", index=False)
    logger.info(f"Saved CelebA splits. Train: {len(train)}")

def process_segmentation_dataset():
    logger.info("Unifying Segmentation Dataset...")
    seg_root = DATA_ROOT / "clothing-segmentation"
    images_dir = seg_root / "images"
    masks_dir = seg_root / "labels" / "pixel_level_labels_colored"
    
    if not images_dir.exists() or not masks_dir.exists():
        logger.warning(f"Segmentation directories not found: {images_dir} or {masks_dir}. Skipping.")
        return
        
    data = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Iterate over masks to ensure we only get labeled data
    for mask_file in masks_dir.iterdir():
        if mask_file.suffix.lower() == '.png':
            # Corresponding image
            img_path = images_dir / f"{mask_file.stem}.jpg"
            
            if img_path.exists():
                data.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_file)
                })
                
    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("No segmentation pairs found.")
        return
        
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    
    train.to_csv(PROCESSED_ROOT / "train_segmentation.csv", index=False)
    val.to_csv(PROCESSED_ROOT / "val_segmentation.csv", index=False)
    test.to_csv(PROCESSED_ROOT / "test_segmentation.csv", index=False)
    logger.info(f"Saved Segmentation splits. Train: {len(train)}")

def main():
    PROCESSED_ROOT.mkdir(exist_ok=True)
    label_map = load_label_map()
    
    process_fashion_dataset(label_map)
    process_polyvore_dataset(label_map)
    process_celeba_dataset()
    process_segmentation_dataset()
    
    logger.info("All data unification completed.")

if __name__ == "__main__":
    main()
