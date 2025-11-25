import torch
import torch.onnx
import json
import argparse
import os
from model import FashionMultiTaskModel

def export_model(args):
    device = torch.device("cpu") # Export on CPU usually
    
    # Load Label Map to get dimensions
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
        
    num_categories = len(label_map['unified_categories'])
    num_subcategories = len(label_map['unified_subcategories'])
    # These need to match what was used in training. 
    # Ideally, these counts should be saved in the checkpoint or a config file.
    # For now, we'll assume we can derive them or pass them in.
    # A robust way is to save a 'model_config.json' during training.
    
    # Placeholder counts if not strictly defined in label_map
    num_colors = 10 # Example, replace with actual
    num_seasons = 4
    num_usages = 5
    
    # Initialize Model
    model = FashionMultiTaskModel(
        num_categories=num_categories,
        num_subcategories=num_subcategories,
        num_colors=num_colors,
        num_seasons=num_seasons,
        num_usages=num_usages
    )
    
    # Load Checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Exporting initialized model.")
        
    model.eval()
    
    # Dummy Input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = os.path.join(args.output_dir, "fashion_model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['category', 'subcategory', 'color', 'season', 'usage', 'embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'category': {0: 'batch_size'}}
    )
    
    print(f"Model exported to {onnx_path}")
    
    # Export Label Map for Flutter
    flutter_map_path = os.path.join(args.output_dir, "labels.json")
    with open(flutter_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
        
    print(f"Label map exported to {flutter_map_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="exported_models")
    parser.add_argument("--label_map", type=str, default="label_map.json")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    export_model(args)
