import argparse
import logging
import torch
import torch.optim as optim
import os
from train import main as train_main, train_classification, train_appearance, train_compatibility, get_transforms
from model import FashionMultiTaskModel
from dataset import FashionDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quality Targets
TARGET_ACC_CLASS = 0.99
TARGET_ACC_APPEAR = 0.99
TARGET_LOSS_COMPAT = 0.02 # Lower is better, but we use negative return value

MAX_RETRIES = 150

def auto_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Initialize Model
    temp_transform, _ = get_transforms()
    temp_dataset = FashionDataset(os.path.join(args.data_dir, "train_fashion.csv"), args.label_map, transform=temp_transform)
    
    model = FashionMultiTaskModel(
        num_categories=temp_dataset.num_categories,
        num_subcategories=10,
        num_colors=temp_dataset.num_colors,
        num_seasons=temp_dataset.num_seasons,
        num_usages=temp_dataset.num_usages,
        num_attributes=40
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Phase 1: Classification ---
    if args.phase == 'all' or args.phase == 'classification':
        logger.info(">>> Checking Classification Quality...")
        current_acc = 0.0
        retries = 0
        
        while current_acc < TARGET_ACC_CLASS and retries < MAX_RETRIES:
            logger.info(f"Starting Classification Training (Attempt {retries+1}/{MAX_RETRIES})")
            current_acc = train_classification(args, model, device, optimizer)
            logger.info(f"Classification Result: {current_acc:.4f} (Target: {TARGET_ACC_CLASS})")
            
            if current_acc < TARGET_ACC_CLASS:
                logger.warning("Target not met. Retraining with lower learning rate...")
                # Decay LR for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                retries += 1
            else:
                logger.info("Classification Target MET!")
                
    # --- Phase 2: Appearance ---
    if args.phase == 'all' or args.phase == 'appearance':
        # Load best classification model if we are continuing
        if args.phase == 'all' and os.path.exists(os.path.join(args.checkpoint_dir, "model_phase1_class.pth")):
             model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_phase1_class.pth")))
             
        logger.info(">>> Checking Appearance Quality...")
        current_acc = 0.0
        retries = 0
        
        # Reset LR for new phase? Or keep decayed? Let's reset to a safe value.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1 # Start smaller for 2nd phase
            
        while current_acc < TARGET_ACC_APPEAR and retries < MAX_RETRIES:
            logger.info(f"Starting Appearance Training (Attempt {retries+1}/{MAX_RETRIES})")
            current_acc = train_appearance(args, model, device, optimizer)
            logger.info(f"Appearance Result: {current_acc:.4f} (Target: {TARGET_ACC_APPEAR})")
            
            if current_acc < TARGET_ACC_APPEAR:
                logger.warning("Target not met. Retraining...")
                retries += 1
            else:
                logger.info("Appearance Target MET!")

    # --- Phase 3: Compatibility ---
    if args.phase == 'all' or args.phase == 'compatibility':
        if args.phase == 'all' and os.path.exists(os.path.join(args.checkpoint_dir, "model_phase2_appear.pth")):
             model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model_phase2_appear.pth")))

        logger.info(">>> Checking Compatibility Quality...")
        # Note: train_compatibility returns negative loss
        current_score = -float('inf') 
        retries = 0
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1
            
        while current_score < -TARGET_LOSS_COMPAT and retries < MAX_RETRIES:
            logger.info(f"Starting Compatibility Training (Attempt {retries+1}/{MAX_RETRIES})")
            current_score = train_compatibility(args, model, device, optimizer)
            # current_score is -loss, so we want it to be > -0.2 (e.g. -0.1)
            logger.info(f"Compatibility Result (Neg Loss): {current_score:.4f} (Target > {-TARGET_LOSS_COMPAT})")
            
            if current_score < -TARGET_LOSS_COMPAT:
                logger.warning("Target not met. Retraining...")
                retries += 1
            else:
                logger.info("Compatibility Target MET!")
                
    logger.info("Auto-Training Complete. Final model ready in checkpoints/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--label_map", type=str, default="label_map.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--phase", type=str, default="all")
    parser.add_argument("--epochs_class", type=int, default=20) # Default per attempt
    parser.add_argument("--epochs_appear", type=int, default=20)
    parser.add_argument("--epochs_compat", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    auto_train(args)
