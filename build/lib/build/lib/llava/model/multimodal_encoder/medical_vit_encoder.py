import sys
import os

headct_path = '/gpfs/data/proteomics/home/bm3772/headCT_foundation'
if headct_path not in sys.path:
    sys.path.insert(0, headct_path)

import torch
import torch.nn as nn
from typing import Optional

from src.models.vit import ViT


class MedicalViTTower(nn.Module):
    """Wrapper for your pretrained medical ViT to work with LLaVA"""
    
    def __init__(
        self, 
        vit_checkpoint_path: str,
        use_all_tokens: bool = True
    ):
        super().__init__()
        
        # Define your ViT architecture (same as your notebook)
        model_params = {
            "img_size": 96,
            "patch_size": 12,
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "in_chans": 3,
            "dropout_rate": 0.0,
            "spatial_dims": 3,
            "patch_embed": 'conv',
            "pos_embed": "sincos",
            "classification": False,
            "num_classes": 2,
            "qkv_bias": False,
            "norm_layer": nn.LayerNorm,
            "post_activation": "Tanh",
        }
        
        # Initialize your ViT
        self.vit = ViT(**model_params)
        
        # Load pretrained weights
        print(f"Loading medical ViT weights from: {vit_checkpoint_path}")
        state_dict = torch.load(vit_checkpoint_path, map_location='cpu', weights_only=True)
        
        # Clean state dict keys (same as your notebook)
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v 
            for k, v in state_dict.items()
        }
        
        # Load weights
        msg = self.vit.load_state_dict(state_dict, strict=False)
        print(f"Loaded ViT weights: {msg}")
        
        # Freeze the ViT - we don't train it
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Configuration
        self.use_all_tokens = use_all_tokens
        self.hidden_size = 768  # Your ViT output dimension
        
        if use_all_tokens:
            self.num_tokens = 513  # 1 [CLS] + 512 patches
        else:
            self.num_tokens = 1    # Just [CLS]
    
    @torch.no_grad()  # No gradients needed - ViT is frozen
    def forward(self, ct_scans):
        """
        Args:
            ct_scans: [batch, 3, 96, 96, 96] - CT scans (3 windowed channels)
        
        Returns:
            features: [batch, num_tokens, 768] - Visual features
        """
        # Extract features from ViT
        features, _ = self.vit(ct_scans)  # [batch, 513, 768]
        
        if self.use_all_tokens:
            # Return all tokens (CLS + patches)
            return features  # [batch, 513, 768]
        else:
            # Return only [CLS] token
            return features[:, 0:1, :]  # [batch, 1, 768]
    
    @property
    def dtype(self):
        return self.vit.patch_embedding.patch_embeddings.weight.dtype
    
    @property
    def device(self):
        return self.vit.patch_embedding.patch_embeddings.weight.device
