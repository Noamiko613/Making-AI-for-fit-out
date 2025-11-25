import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
import logging

from model import FashionMultiTaskModel
from dataset import FashionDataset, CelebADataset, PolyvoreDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def calculate_accuracy(outputs, labels, task='classification'):
    if task == 'classification':
        # Average accuracy across all heads
        acc_cat = (outputs['category'].argmax(1) == labels['category']).float().mean()
        acc_col = (outputs['color'].argmax(1) == labels['color']).float().mean()
        acc_seas = (outputs['season'].argmax(1) == labels['season']).float().mean()
        acc_use = (outputs['usage'].argmax(1) == labels['usage']).float().mean()
        return (acc_cat + acc_col + acc_seas + acc_use) / 4.0
    elif task == 'appearance':
        # Multi-label accuracy (threshold 0.5)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        return (preds == labels).float().mean()
    elif task == 'compatibility':
        # For embedding loss, accuracy is tricky. We'll use Loss as the metric (lower is better).
        # Returning -Loss so that "higher is better" logic still works for the auto-trainer.
        return -outputs # outputs here is actually the loss value passed in

def train_classification(args, model, device, optimizer):
    logger.info("--- Starting Phase 1: Classification Training ---")
    train_transform, val_transform = get_transforms()
    
    train_dataset = FashionDataset(os.path.join(args.data_dir, "train_fashion.csv"), args.label_map, transform=train_transform)
    val_dataset = FashionDataset(os.path.join(args.data_dir, "val_fashion.csv"), args.label_map, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_color = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_season = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_usage = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs_class):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Class Epoch {epoch+1}")
        for batch in pbar:
            images = batch['image'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, task='classification')
            
            loss = criterion_cat(outputs['category'], batch['category'].to(device)) + \
                   criterion_color(outputs['color'], batch['color'].to(device)) + \
                   criterion_season(outputs['season'], batch['season'].to(device)) + \
                   criterion_usage(outputs['usage'], batch['usage'].to(device))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                outputs = model(images, task='classification')
                
                # Prepare labels dict for accuracy helper
                labels = {
                    'category': batch['category'].to(device),
                    'color': batch['color'].to(device),
                    'season': batch['season'].to(device),
                    'usage': batch['usage'].to(device)
                }
                val_acc += calculate_accuracy(outputs, labels, task='classification').item()
        
        val_acc /= len(val_loader)
        logger.info(f"Epoch {epoch+1} Classification Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_phase1_class.pth"))
            
    return best_val_acc

def train_appearance(args, model, device, optimizer):
    logger.info("--- Starting Phase 2: Appearance Training ---")
    train_transform, val_transform = get_transforms()
    
    train_dataset = CelebADataset(os.path.join(args.data_dir, "train_celeba.csv"), transform=train_transform)
    # Use subset of train for validation if val set not explicitly passed, or split manually.
    # For simplicity, we'll just use the train set metrics or split a small val set.
    # Let's use the val_celeba.csv we created.
    val_dataset = CelebADataset(os.path.join(args.data_dir, "val_celeba.csv"), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss() 
    best_val_acc = 0.0
    
    for epoch in range(args.epochs_appear):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Appear Epoch {epoch+1}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['attributes'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, task='appearance')
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['attributes'].to(device)
                outputs = model(images, task='appearance')
                val_acc += calculate_accuracy(outputs, labels, task='appearance').item()
                
        val_acc /= len(val_loader)
        logger.info(f"Epoch {epoch+1} Appearance Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_phase2_appear.pth"))
            
    return best_val_acc

def train_compatibility(args, model, device, optimizer):
    logger.info("--- Starting Phase 3: Compatibility Training ---")
    train_transform, val_transform = get_transforms()
    
    train_dataset = PolyvoreDataset(os.path.join(args.data_dir, "train_polyvore.json"), transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    criterion = nn.CosineEmbeddingLoss()
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs_compat):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Compat Epoch {epoch+1}")
        for batch in pbar:
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            target = batch['label'].to(device) 
            
            optimizer.zero_grad()
            emb1 = model(img1, task='compatibility')
            emb2 = model(img2, task='compatibility')
            
            loss = criterion(emb1, emb2, target)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        avg_loss = running_loss/len(train_loader)
        logger.info(f"Epoch {epoch+1} Compatibility Loss: {avg_loss:.4f}")
        
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_phase3_compat.pth"))
            
    return -best_val_loss # Return negative loss so higher is better

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset first to get dimensions
    # We use the training set for this as it should contain all classes
    temp_transform, _ = get_transforms()
    temp_dataset = FashionDataset(os.path.join(args.data_dir, "train_fashion.csv"), args.label_map, transform=temp_transform)
    
    # Initialize Model with correct dimensions
    model = FashionMultiTaskModel(
        num_categories=temp_dataset.num_categories,
        num_subcategories=10, # Placeholder, not used in current loss
        num_colors=temp_dataset.num_colors,
        num_seasons=temp_dataset.num_seasons,
        num_usages=temp_dataset.num_usages,
        num_attributes=40 # CelebA has 40 attributes
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.phase == 'all' or args.phase == 'classification':
        train_classification(args, model, device, optimizer)
        
    if args.phase == 'all' or args.phase == 'appearance':
        # Load best from previous phase if sequential
        if args.phase == 'all':
            # In a real run, we'd load the best checkpoint. 
            # For now, we continue with the current model state.
            pass
        train_appearance(args, model, device, optimizer)
        
    if args.phase == 'all' or args.phase == 'compatibility':
        if args.phase == 'all':
             pass
        train_compatibility(args, model, device, optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--label_map", type=str, default="label_map.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--phase", type=str, default="all", choices=['all', 'classification', 'appearance', 'compatibility'])
    parser.add_argument("--epochs_class", type=int, default=5)
    parser.add_argument("--epochs_appear", type=int, default=2)
    parser.add_argument("--epochs_compat", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)
