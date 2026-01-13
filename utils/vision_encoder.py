####################################################
# Author: fvilmos
# https://github.com/fvilmos
####################################################

import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    Use CNN as backbone
    """
    def __init__(self):
        super().__init__()

        # import only with class
        from torchvision.models import resnet18, ResNet18_Weights
        cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # B,C,H,W = 512,7,7, get just the features
        net = torch.nn.Sequential(*list(cnn.children())[:-2])
        self.net = net
        
    def forward(self, x):
        x = self.net(x).flatten(2).permute(0, 2, 1)
        # B, HxW, C
        return x

class ViTEncoder(nn.Module):
    """
    Use ViT as backbone
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import vit_b_32, ViT_B_32_Weights
        vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        
        self.vit = vit
       
    def forward(self, x):
        x = self.vit._process_input(x)
        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.encoder(x)
        # (B, N+1, D)
        return x
       
class VisionProjection(nn.Module):
    def __init__(self, in_dim, out_dim, vis_hxw_out):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # add positional encoding as learnable param
        self.pos_encoder = nn.Parameter(torch.randn(1, vis_hxw_out, out_dim))

    def forward(self, x):
        x = self.norm(self.proj(x))
        x = x + self.pos_encoder
        return x