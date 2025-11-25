import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import json
import random

class FashionDataset(Dataset):
    """
    For Classification: Category, Color, Season, Usage
    """
    def __init__(self, csv_file, label_map_path, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
            
        self.cat_to_idx = {name: i for i, name in enumerate(self.label_map['unified_categories'])}
        
        # Dynamic mappings for attributes (simplified for now)
        self.colors = sorted(self.data['color'].dropna().unique().tolist())
        self.color_to_idx = {name: i for i, name in enumerate(self.colors)}
        
        self.seasons = sorted(self.data['season'].dropna().unique().tolist())
        self.season_to_idx = {name: i for i, name in enumerate(self.seasons)}
        
        self.usages = sorted(self.data['usage'].dropna().unique().tolist())
        self.usage_to_idx = {name: i for i, name in enumerate(self.usages)}

    @property
    def num_categories(self):
        return len(self.label_map['unified_categories'])

    @property
    def num_colors(self):
        return len(self.colors)

    @property
    def num_seasons(self):
        return len(self.seasons)

    @property
    def num_usages(self):
        return len(self.usages)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return {
            'type': 'fashion',
            'image': image,
            'category': torch.tensor(self.cat_to_idx.get(row['category'], -100), dtype=torch.long),
            'color': torch.tensor(self.color_to_idx.get(row['color'], -100), dtype=torch.long),
            'season': torch.tensor(self.season_to_idx.get(row['season'], -100), dtype=torch.long),
            'usage': torch.tensor(self.usage_to_idx.get(row['usage'], -100), dtype=torch.long)
        }

class PolyvoreDataset(Dataset):
    """
    For Compatibility: Returns a pair of items (compatible) or random pair (incompatible).
    """
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.outfits = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.outfits)

    def __getitem__(self, idx):
        outfit = self.outfits[idx]
        items = outfit['items']
        
        # Positive pair (from same outfit)
        if len(items) < 2:
            return self.__getitem__((idx + 1) % len(self))
            
        item1, item2 = random.sample(items, 2)
        
        try:
            img1 = Image.open(item1['image_path']).convert('RGB')
            img2 = Image.open(item2['image_path']).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return {
            'type': 'polyvore',
            'image1': img1,
            'image2': img2,
            'label': torch.tensor(1.0, dtype=torch.float) # Compatible
        }

class CelebADataset(Dataset):
    """
    For Appearance: Multi-label attributes (Skin, Hair, etc.)
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.attr_cols = [c for c in self.data.columns if c not in ['image_id', 'image_path']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # Attributes: 1 (present), -1 (absent) -> Map to 1, 0
        attrs = row[self.attr_cols].values.astype(float)
        attrs = (attrs + 1) / 2 # Convert -1/1 to 0/1
        
        return {
            'type': 'celeba',
            'image': image,
            'attributes': torch.tensor(attrs, dtype=torch.float)
        }

class SegmentationDataset(Dataset):
    """
    For Body Sizing/Segmentation: Image + Mask
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform # Needs careful handling for masks (resize nearest)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            image = Image.open(row['image_path']).convert('RGB')
            mask = Image.open(row['mask_path']) # Keep as is (palette or L)
        except Exception:
            image = Image.new('RGB', (224, 224))
            mask = Image.new('L', (224, 224))
            
        if self.transform:
            # Note: Standard transforms might mess up masks (interpolation).
            # Ideally use functional transforms with same params.
            # For now, we just resize.
            image = self.transform(image)
            # Mask resize should be Nearest Neighbor
            # We'll handle mask resize manually if needed or assume transform handles it
            pass
            
        return {
            'type': 'segmentation',
            'image': image,
            # 'mask': mask # Return raw PIL or tensor
        }
