import torch
import torch.nn as nn
import torchvision.models as models

class FashionMultiTaskModel(nn.Module):
    def __init__(self, num_categories, num_subcategories, num_colors, num_seasons, num_usages, num_attributes, embedding_dim=128):
        super(FashionMultiTaskModel, self).__init__()
        
        # Backbone: MobileNetV3 Large (Pretrained)
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        self.feature_extractor = self.backbone.features
        self.avgpool = self.backbone.avgpool
        self.in_features = 960
        
        # --- Classification Heads (Fashion Dataset) ---
        self.category_head = nn.Sequential(
            nn.Linear(self.in_features, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_categories)
        )
        self.color_head = nn.Sequential(
            nn.Linear(self.in_features, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_colors)
        )
        self.season_head = nn.Sequential(
            nn.Linear(self.in_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_seasons)
        )
        self.usage_head = nn.Sequential(
            nn.Linear(self.in_features, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_usages)
        )
        
        # --- Compatibility Head (Polyvore Dataset) ---
        # Outputs an embedding vector. We use Euclidean distance or Cosine similarity in loss.
        self.embedding_head = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # --- Appearance Head (CelebA Dataset) ---
        # Multi-label classification (Sigmoid activation in loss, so Linear here)
        self.appearance_head = nn.Sequential(
            nn.Linear(self.in_features, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_attributes) 
        )
        
    def forward(self, x, task='classification'):
        # Shared Features
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if task == 'classification':
            return {
                'category': self.category_head(x),
                'color': self.color_head(x),
                'season': self.season_head(x),
                'usage': self.usage_head(x)
            }
        elif task == 'compatibility':
            return self.embedding_head(x)
        elif task == 'appearance':
            return self.appearance_head(x)
        else:
            # Return everything (inference)
            return {
                'category': self.category_head(x),
                'color': self.color_head(x),
                'season': self.season_head(x),
                'usage': self.usage_head(x),
                'embedding': self.embedding_head(x),
                'appearance': self.appearance_head(x)
            }

    def get_embedding(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.embedding_head(x)
