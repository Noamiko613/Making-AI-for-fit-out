#!/usr/bin/env python3

"""
ai_designer_pipeline.py

All-in-one pipeline:
attempt to download automatable fashion datasets (Kaggle / GitHub / HuggingFace)
expect manual placement for restricted datasets (DeepFashion / DeepFashion2)
preprocess, normalize, extract colors, optional segmentation
build PyTorch datasets
train a multi-head model (classification + multi-label + embedding InfoNCE)
save checkpoints and export TorchScript
USAGE:
python ai_designer_pipeline.py --download --preprocess --train
NOTES:
For Kaggle downloads set env var KAGGLE_USERNAME and KAGGLE_KEY or place kaggle.json.
For HuggingFace downloads set HUGGINGFACE_TOKEN if needed.
Place manual-download datasets under ai_designer_data/raw/public_datasets/<dataset_name>/
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import random
import math
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Basic dependencies. Install with:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install pillow opencv-python pandas tqdm kaggle requests torchvision albumentations sklearn
# optional: pip install onnx onnxruntime
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from torchvision.io import read_image
except Exception as e:
    print("Missing PyTorch / torchvision. Install requirements as shown in header.")
    raise e

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
from tqdm import tqdm
import requests
from sklearn.cluster import KMeans

# Pandas error for reading empty files
try:
    from pandas.errors import EmptyDataError
except ImportError:
    class EmptyDataError(Exception):
        pass

# ---------- Config ----------
ROOT = Path.cwd() / "ai_designer_data"
RAW = ROOT / "raw"
PUBLIC = RAW / "public_datasets"
USER = RAW / "user_uploads"
PROCESSED = ROOT / "processed"
IMAGES_OUT = PROCESSED / "images"
MASKS_OUT = PROCESSED / "masks"
LABELS_CSV = PROCESSED / "labels.csv"
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
CHECKPOINTS = MODELS_DIR / "checkpoints"

IMG_SIZE = 256  # on-device-friendly size; change to 384 if you prefer higher-res
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Running on device:", DEVICE)

# ---------- Utilities ----------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def ensure_dirs():
    for p in [RAW, PUBLIC, USER, PROCESSED, IMAGES_OUT, MASKS_OUT, DATASETS_DIR, MODELS_DIR, CHECKPOINTS]:
        p.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# ---------- Dataset download helpers ----------
def run_cmd(cmd: List[str], cwd: Optional[str]=None):
    """Run shell command and stream output."""
    print("RUN:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def download_from_github_raw(url: str, dest: Path):
    """Download file via direct URL to raw content."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def kaggle_download(dataset: str, dest: Path):
    """Download a dataset from Kaggle if kaggle API available and credentials set."""
    # expects "username/dataset-name" style for API
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        print("kaggle CLI not found; skipping Kaggle download. Install with pip install kaggle and configure credentials.")
        return False
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    run_cmd(cmd)
    return True

def hf_download(repo_id: str, filename: Optional[str], dest: Path, token_env="HUGGINGFACE_TOKEN"):
    """Download file from Hugging Face repo raw link if possible (may require token)."""
    token = os.environ.get(token_env)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}" if filename else f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=headers, stream=True, timeout=30)

    if r.status_code != 200:
        print(f"HuggingFace download failed: {r.status_code} {url}")
        return False

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return True

# ---------- Known dataset download mapping ----------
# Notes: some dataset downloads require manual steps (DeepFashion2 form). The script will attempt Kaggle/GitHub/HF for mirrors.
DATASET_SOURCES = {
    "fashion200k": {
        "type": "github",
        "note": "Fashion200k mirror on GitHub; contains image_urls.txt; script will download images from URLs.",
        "urls": ["https://raw.githubusercontent.com/xthan/fashion-200k/master/image_urls.txt"],
        "dest": PUBLIC / "fashion200k"
    },
    "polyvore_outfits": {
        "type": "github",
        "note": "Polyvore outfits dataset JSON on GitHub.",
        "urls": [
            "https://raw.githubusercontent.com/xthan/polyvore-dataset/master/train_no_dup.json",
            "https://raw.githubusercontent.com/xthan/polyvore-dataset/master/valid_no_dup.json",
            "https://raw.githubusercontent.com/xthan/polyvore-dataset/master/test_no_dup.json"
        ],
        "dest": PUBLIC / "polyvore_outfits"
    },
    "modanet": {
        "type": "github",
        "note": "ModaNet annotations / images. Large; choose the subset. If download fails, use manual Google Drive mirror.",
        "urls": ["https://raw.githubusercontent.com/eBay/modanet/master/README.md"],
        "dest": PUBLIC / "modanet"
    },
    "fashionpedia": {
        "type": "github",
        "note": "Fashionpedia dataset and annotation files available via project page/GitHub.",
        "urls": ["https://raw.githubusercontent.com/cvdfoundation/fashionpedia/main/README.md"],
        "dest": PUBLIC / "fashionpedia"
    },
    "deepfashion2": {
        "type": "manual",
        "note": "DeepFashion2 requires dataset agreement / form and sometimes password. Please download manually and place extracted folder at: ai_designer_data/raw/public_datasets/deepfashion2/",
        "dest": PUBLIC / "deepfashion2"
    },
    "deepfashion": {
        "type": "manual",
        "note": "Original DeepFashion has multiple releases. Please download and place in ai_designer_data/raw/public_datasets/deepfashion/",
        "dest": PUBLIC / "deepfashion"
    },
    "kaggle_fashion_products": {
        "type": "kaggle",
        "note": "Kaggle product images dataset (require kaggle api credentials).",
        "dataset": "paramaggarwal/fashion-product-images-dataset",
        "dest": PUBLIC / "kaggle_fashion_products"
    }
}

def attempt_download_all(force=False):
    print("Attempting to download automatable datasets (where possible).")
    for name, meta in DATASET_SOURCES.items():
        dest = meta["dest"]
        if dest.exists() and not force:
            print(f"{name} already present at {dest} — skipping (use --force to re-download).")
            continue

        print(f"Processing dataset: {name} -> {dest}")

        if meta["type"] == "kaggle":
            success = kaggle_download(meta["dataset"], dest)
            if not success:
                print(f"Kaggle download failed or not configured. Please download {meta['dataset']} manually and place under {dest}")

        elif meta["type"] == "github":
            dest.mkdir(parents=True, exist_ok=True)
            for url in meta.get("urls", []):
                try:
                    filename = url.split("/")[-1]
                    target = dest / filename
                    print(f"Downloading {url} -> {target}")
                    download_from_github_raw(url, target)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
            print("If you cannot download via GitHub raw links, please download manually and place files under", dest)

        elif meta["type"] == "manual":
            print(f"{name} requires manual download. See note: {meta['note']}")
            dest.mkdir(parents=True, exist_ok=True)

    print("Download attempts finished. Please check logs for any manual actions required.")

# ---------- Preprocessing helpers ----------
def pil_open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def resize_and_normalize(img: Image.Image, size=IMG_SIZE) -> Image.Image:
    # center-crop to preserve aspect or do letterbox? We'll use center-crop then resize
    w, h = img.size
    min_wh = min(w, h)
    left = (w - min_wh) // 2
    top = (h - min_wh) // 2
    img = img.crop((left, top, left + min_wh, top + min_wh))
    img = img.resize((size, size), Image.BILINEAR)
    return img

def extract_primary_color(img: Image.Image, k=3) -> Tuple[int, int, int]:
    # simple k-means on pixels to get dominant color
    arr = np.array(img).reshape(-1, 3).astype(float)

    # sample to speed up
    if arr.shape[0] > 2000:
        idx = np.random.choice(arr.shape[0], size=2000, replace=False)
        arr_s = arr[idx]
    else:
        arr_s = arr

    # k-means (simple)
    km = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(arr_s)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[np.argmax(counts)]
    return tuple([int(x) for x in dominant])

# ---------- Universal ingestion & label builder ----------
def ingest_public_images_to_processed():
    """Walk public datasets and bring images into processed/images// and create initial labels rows."""
    rows = []

    # scan known dataset folders under PUBLIC
    for ds in PUBLIC.iterdir():
        if not ds.is_dir():
            continue

        # special handling: fashion200k image_urls
        if ds.name == "fashion200k" and (ds / "image_urls.txt").exists():
            print("Processing fashion200k urls...")
            with open(ds / "image_urls.txt", "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="fashion200k urls"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    name, url = parts[0], parts[1]
                    out_img = IMAGES_OUT / "fashion200k" / name
                    out_img.parent.mkdir(parents=True, exist_ok=True)

                    if not out_img.exists():
                        try:
                            download_from_github_raw(url, out_img)
                        except Exception:
                            # fallback: skip
                            continue
                    rows.append({"image_path": str(out_img), "source": "fashion200k"})

        elif ds.name == "polyvore_outfits":
            # polyvore provides JSONs referencing images - some links are remote; we will copy JSON files and later attempt to download images referenced
            for fname in ["train_no_dup.json", "valid_no_dup.json", "test_no_dup.json"]:
                fpath = ds / fname
                if fpath.exists():
                    tgt = IMAGES_OUT / "polyvore_outfits" / fname
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(fpath), str(tgt))

        else:
            # generic: copy image files (jpg/png) into processed images
            for root, _, files in os.walk(str(ds)):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        src = Path(root) / f
                        rel = src.relative_to(PUBLIC)
                        dest = IMAGES_OUT / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            try:
                                shutil.copy(src, dest)
                            except Exception:
                                # some files may be links; skip
                                continue
                        rows.append({"image_path": str(dest), "source": ds.name})

    # user uploads
    for root, _, files in os.walk(str(USER)):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src = Path(root) / f
                dest = IMAGES_OUT / "user_uploads" / f
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy(src, dest)
                rows.append({"image_path": str(dest), "source": "user_uploads"})

    print(f"Ingested approx {len(rows)} images.")

    # Save initial labels csv (many labels will be empty and filled later via auto-tagging)
    df = pd.DataFrame(rows)
    combined = df # Start with the newly ingested data

    if LABELS_CSV.exists():
        try:
            # Attempt to read existing data
            existing = pd.read_csv(LABELS_CSV)
            # Combine existing and new data, dropping duplicates
            combined = pd.concat([existing, df]).drop_duplicates(subset=["image_path"]).reset_index(drop=True)
        except EmptyDataError:
            # If the existing file is empty or only has headers, treat 'combined' as just the new 'df'
            print(f"Warning: Existing labels file {LABELS_CSV.name} was empty or invalid. Overwriting with new data.")
        except Exception as e:
            # Handle other potential read errors (e.g., corrupted file)
            print(f"Error reading existing labels file: {e}. Overwriting with new data.")
            
    # Always save the combined DataFrame (new or combined existing + new)
    combined.to_csv(LABELS_CSV, index=False)

    print("Labels CSV updated:", LABELS_CSV)

# ---------- Auto-tagging (basic) ----------
def auto_tag_images(num_samples=None):
    """
    A simple auto-tagging pass:
    - Resize + save as normalized images
    - Extract primary color
    - Optionally run a small pretrained model (if available) to predict category (placeholder)
    """
    df = pd.read_csv(LABELS_CSV)
    total = len(df) if num_samples is None else min(num_samples, len(df))
    print("Auto-tagging", total, "images")

    for i, row in tqdm(df.head(total).iterrows(), total=total):
        img_path = Path(row["image_path"])
        if not img_path.exists():
            continue

        try:
            img = pil_open_rgb(img_path)
            img_r = resize_and_normalize(img, size=IMG_SIZE)

            # save normalized preview
            # NOTE: The original code had a confusing replacement logic which I've streamlined/corrected to save the processed image back to its original location (or ensure the original image path points to the normalized version for the dataset).
            # For simplicity and to match the original intent of 'processed/images/', we assume the image is processed in place if this is intended as a final normalized storage.
            savep = img_path
            savep.parent.mkdir(parents=True, exist_ok=True)
            img_r.save(savep)

            # color
            color = extract_primary_color(img_r)
            df.at[i, "primary_color_rgb"] = json.dumps(color)

            # placeholder: empty category/style/pattern columns for manual labeling or future model predicted tags
            for c in ["category", "pattern", "style", "season", "colors_multi"]:
                if c not in df.columns:
                    df[c] = ""

        except Exception as e:
            print("Error processing", img_path, e)
            continue

    df.to_csv(LABELS_CSV, index=False)
    print("Auto-tagging finished. Updated", LABELS_CSV)

# ---------- Simple PyTorch Dataset ----------
class FashionSimpleDataset(Dataset):
    def __init__(self, labels_csv: str, transform=None, mode="train"):
        self.df = pd.read_csv(labels_csv)
        # drop rows with missing image
        self.df = self.df[self.df["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        self.transform = transform

        # create dummy maps for category/style/pattern if not present
        self.category_map = {"unknown": 0}
        self.style_map = {"unknown": 0}
        self.pattern_map = {"unknown": 0}

        # try to build maps
        if "category" in self.df.columns:
            uniques = set(self.df["category"].dropna().astype(str).unique())
            uniques = [u for u in uniques if u not in ("", "nan")]
            for i, u in enumerate(sorted(uniques)):
                if u not in self.category_map:
                    self.category_map[u] = len(self.category_map)

        if "style" in self.df.columns:
            uniques = set(self.df["style"].dropna().astype(str).unique())
            uniques = [u for u in uniques if u not in ("", "nan")]
            for i, u in enumerate(sorted(uniques)):
                if u not in self.style_map:
                    self.style_map[u] = len(self.style_map)

        if "pattern" in self.df.columns:
            uniques = set(self.df["pattern"].dropna().astype(str).unique())
            uniques = [u for u in uniques if u not in ("", "nan")]
            for i, u in enumerate(sorted(uniques)):
                if u not in self.pattern_map:
                    self.pattern_map[u] = len(self.pattern_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(resize_and_normalize(img, size=IMG_SIZE))

        # Label handling: best-effort mapping
        cat = r.get("category", "") if "category" in r else ""
        cat_idx = self.category_map.get(str(cat), 0)

        style = r.get("style", "") if "style" in r else ""
        style_idx = self.style_map.get(str(style), 0)

        pat = r.get("pattern", "") if "pattern" in r else ""
        pat_idx = self.pattern_map.get(str(pat), 0)

        # For multi-label colors we don't handle full mapping here; use dummy zeros
        colors_vec = np.zeros(10, dtype=np.float32)

        return {"image": img, "category": cat_idx, "style": style_idx, "pattern": pat_idx, "colors": colors_vec}

# ---------- Model definition (multi-head with embedding) ----------
class MultiTaskNet(nn.Module):
    def __init__(self, embedding_dim=256, n_category=10, n_colors=10, n_pattern=8, n_style=6, n_season=4):
        super().__init__()

        # Use a lightweight backbone: MobileNetV3 small or resnet18 if not available
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            feat_dim = backbone.classifier[0].in_features
        except Exception:
            try:
                # Use hub for resnet18 if torchvision.models failed or is older
                res = torch.hub.load('pytorch/vision:v0.15.2', 'resnet18', pretrained=True)
                self.features = nn.Sequential(*list(res.children())[:-1])
                feat_dim = res.fc.in_features
            except Exception:
                # Fallback to a simple placeholder if all else fails
                print("Warning: Could not load MobileNetV3 or ResNet18. Using placeholder features.")
                feat_dim = 512 # Placeholder
                self.features = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(feat_dim, embedding_dim)
        self.category_head = nn.Linear(feat_dim, n_category)
        self.color_head = nn.Linear(feat_dim, n_colors)
        self.pattern_head = nn.Linear(feat_dim, n_pattern)
        self.style_head = nn.Linear(feat_dim, n_style)
        self.season_head = nn.Linear(feat_dim, n_season)

    def forward(self, x):
        # features may expect NCHW
        # Pass through identity if feature extraction failed (placeholder)
        f_feat_map = self.features(x)
        f = self.pool(f_feat_map).reshape(x.size(0), -1)

        emb = self.embedding(f)
        emb = nn.functional.normalize(emb, dim=1)

        return {
            "embedding": emb,
            "category": self.category_head(f),
            "colors": self.color_head(f),
            "pattern": self.pattern_head(f),
            "style": self.style_head(f),
            "season": self.season_head(f)
        }

# ---------- Contrastive loss (InfoNCE) ----------
class InfoNCEContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, embeddings: torch.Tensor):
        """
        embeddings: (batch_size, embed_dim) where batch is organized as pairs [a1,a2,b1,b2,...]
        For simplicity we will compute sim matrix and use self-sim positives on pairs with offset 1 when batch arranged in pairs.
        This is a minimal implementation for demonstration.
        """
        device = embeddings.device
        bs = embeddings.shape[0]

        # Calculate similarity matrix (logits)
        sim = torch.matmul(embeddings, embeddings.T) / self.temp

        loss = 0.0
        count = 0

        # Iterate over pairs (i, i+1) where i is even
        for i in range(0, bs, 2):
            if i + 1 >= bs:
                break

            # 1. Loss for i, where i+1 is positive
            logits1 = sim[i]
            # mask out self
            logits1[i] = -1e9
            # Cross entropy loss where target is the index of the positive (i+1)
            l1 = nn.functional.cross_entropy(logits1.unsqueeze(0), torch.tensor([i + 1], device=device))
            loss += l1

            # 2. Loss for i+1, where i is positive
            logits2 = sim[i + 1]
            # mask out self
            logits2[i + 1] = -1e9
            # Cross entropy loss where target is the index of the positive (i)
            l2 = nn.functional.cross_entropy(logits2.unsqueeze(0), torch.tensor([i], device=device))
            loss += l2

            count += 2

        if count == 0:
            return torch.tensor(0.0, device=device)

        return loss / count

# ---------- Training loop ----------
def train_loop(epochs=5, resume_checkpoint=None, export_torchscript=True):
    # transforms
    transform = T.Compose([
        T.Lambda(lambda img: resize_and_normalize(img, size=IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = FashionSimpleDataset(str(LABELS_CSV), transform=transform)

    if len(ds) == 0:
        raise RuntimeError("No processed images found. Run ingest/preprocess first.")

    # For InfoNCE pairing, we will create synthetic pairs by duplicating batch with small augmentations (for simplicity)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    # Initialize model with correct head counts based on dataset maps
    n_category = len(ds.category_map)
    n_style = len(ds.style_map)
    n_pattern = len(ds.pattern_map)
    # n_colors, n_season are fixed (placeholder 10, 4) unless dataset provides them
    model = MultiTaskNet(n_category=n_category, n_style=n_style, n_pattern=n_pattern).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    ce = nn.CrossEntropyLoss()
    # bce = nn.BCEWithLogitsLoss() # Only used if multi-label targets were available
    info_nce = InfoNCEContrastiveLoss()

    start_epoch = 0
    if resume_checkpoint:
        ck = torch.load(resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        start_epoch = ck.get('epoch', 0) + 1
        print("Resumed from checkpoint", resume_checkpoint)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs-1}")

        for batch in pbar:
            imgs = batch["image"].to(DEVICE)

            # create pair-aug: naive pairing by duplicating imgs with mild transforms (here we simply shuffle slightly)
            # The InfoNCE needs pairs of related items. Shuffling is a *very* naive way.
            # A proper implementation would use two different augmentations of the *same* image.
            imgs2 = imgs[torch.randperm(imgs.size(0))]
            combined = torch.cat([imgs, imgs2], dim=0)  # batch*2

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(combined)
                # split outputs
                # classification losses operate only on the first half
                out_first = {k: v[:imgs.size(0)] for k, v in out.items()}

                loss = 0.0
                # category/style/pattern
                cat = batch.get("category", None)
                if cat is not None and len(ds.category_map) > 1:
                    cat = cat.to(DEVICE)
                    loss += ce(out_first["category"], cat) * 1.0

                pat = batch.get("pattern", None)
                if pat is not None and len(ds.pattern_map) > 1:
                    pat = pat.to(DEVICE)
                    loss += ce(out_first["pattern"], pat) * 0.5

                style = batch.get("style", None)
                if style is not None and len(ds.style_map) > 1:
                    style = style.to(DEVICE)
                    loss += ce(out_first["style"], style) * 0.2

                # embedding InfoNCE on full combined batch
                emb = out["embedding"]
                loss += info_nce(emb) * 1.0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / ((pbar.n or 1))})

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }
        torch.save(ckpt, CHECKPOINTS / f"ckpt_epoch_{epoch}.pt")
        print(f"Saved checkpoint epoch {epoch} to {CHECKPOINTS}")

    # export torchscript for mobile (basic trace)
    if export_torchscript:
        model.eval()
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
            traced.save(str(MODELS_DIR / "ai_designer_traced.pt"))
        print("Exported TorchScript model to", MODELS_DIR / "ai_designer_traced.pt")

# ---------- Main orchestration ----------
def main(args):
    ensure_dirs()

    if args.download:
        attempt_download_all(force=args.force)

    if args.ingest:
        ingest_public_images_to_processed()

    if args.autotag:
        auto_tag_images(num_samples=args.autotag_samples)

    if args.train:
        train_loop(epochs=args.epochs, resume_checkpoint=args.resume)

    print("Pipeline finished.")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Attempt to download automatable datasets (Kaggle/GitHub/HF).")
    parser.add_argument("--force", action="store_true", help="Force re-download even if data exists.")
    parser.add_argument("--ingest", action="store_true", help="Ingest images from raw/public_datasets and user_uploads into processed images and build labels.")
    parser.add_argument("--autotag", action="store_true", help="Run simple auto-tagging (color extraction, resizing).")
    parser.add_argument("--autotag-samples", type=int, default=None)
    parser.add_argument("--train", action="store_true", help="Start training using processed labels.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    args = parser.parse_args()
    main(args)