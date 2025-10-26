import torch
import torch.nn as nn # neural network modules (e.g. Conv2d, Linear, Sequential)
import torch.nn.functional as F # activation functions and pooling
from torchvision import models # pre-trained models e.g. ResNet
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms #common image transforms like resizing, cropping, normalization, and converting

class DELGBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(DELGBackbone, self).__init__()  
        resnet = models.resnet50(pretrained=pretrained) #Loads a ResNet-50 model from torchvision.
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Take all conv layers, remove last 2 layer avgpool & fully connected

    def forward(self, x):  # input [Batch size, 3 channel, Height, Width]
        return self.features(x)  #[B, 2048, H/32, W/32] (B=batch, 2048 channels, spatially downsampled 32x).
    
class GeM(nn.Module): 
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        #x.clamp(min=self.eps) prevents values below eps
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p) 
        # This reduces [B, C, H, W] → [B, C, 1, 1] → can flatten to [B, C] → global descriptor for the image. for global retrieval / top-k candidate search.
        # Input: [16, 2048, 7, 7] feature maps (from ResNet backbone)
        # After GeM: [16, 2048, 1, 1] → flatten → [16, 2048]


class AttentionModule(nn.Module):
    def __init__(self, in_channels=2048): # same as the ResNet50 backbone output channels ([B, 2048, H/32, W/32] e.g. [B, 2048, 7, 7]
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1) # Reduces the channel dimension , Keeps the spatial dimension (still 7×7)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1) # Collapses 512 channels into a single attention map per image. This gives [B, 1, 7, 7]
    
    def forward(self, x):
        att = F.relu(self.conv1(x))
        att = torch.sigmoid(self.conv2(att))  # [B,1,H,W] Normalizes output between 0 and 1
        return att

class DELG(nn.Module):
    def __init__(self, pretrained=True, use_global=True, use_local=True):
        super(DELG, self).__init__()
        self.backbone = DELGBackbone(pretrained) # ResNet50 conv layers
        self.gem = GeM() if use_global else None
        self.attention = AttentionModule() if use_local else None
        self.use_global = use_global # allows to switch off global or local descriptors
        self.use_local = use_local # allows to switch off global or local descriptors
    
    def forward(self, x):
        fmap = self.backbone(x)  # Output size [B,2048,H/32,W/32]
        output = {}
        
        if self.use_global: # add global descriptor (GeM pooled)
            gfeat = self.gem(fmap)  # [B,2048,1,1]
            gfeat = gfeat.view(gfeat.size(0), -1)  # flatten [B,2048]
            gfeat = F.normalize(gfeat, p=2, dim=1)  # L2 normalize (good for cosine similarity retrieval)
            output['global'] = gfeat
        
        if self.use_local: # add local descriptors with attention
            att = self.attention(fmap)  # [B,1,H,W]
            local_desc = fmap * att       # attention-weighted descriptors
            B,C,H,W = local_desc.shape
            local_desc = local_desc.view(B,C,H*W).permute(0,2,1)  # Reshape to [B,H*W,2048] each spatial location is a descriptor.
            output['local'] = {'descriptors': local_desc, 'attention': att}
        
        return output
    
