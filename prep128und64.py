#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:04:25 2026

@author: alex
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image


# ============================================================
# 1. Dataset
# ============================================================

class CelebASRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith((".jpg", ".png"))]

        print(len(self.files))
        # --- transforms ---
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])

        img = Image.open(path).convert("RGB")

        # ----------------------------------------------------
        # 1. Center crop (CelebA standard: square face crop)
        # ----------------------------------------------------
        img = TF.center_crop(img, min(img.size))  # square crop

        # ----------------------------------------------------
        # 2. Resize to 128x128 (HR target)
        # ----------------------------------------------------
        img_128 = TF.resize(img, (128, 128), interpolation=Image.BICUBIC)

        # ----------------------------------------------------
        # 3. Downsample to 64x64 (LR input)
        # (use area-like downsampling for better SR training)
        # ----------------------------------------------------
        img_64 = TF.resize(img_128, (64, 64), interpolation=Image.BOX)

        # ----------------------------------------------------
        # 4. Convert to tensor and normalize to [-1, 1]
        # ----------------------------------------------------
        x128 = self.to_tensor(img_128) * 2 - 1
        x64 = self.to_tensor(img_64) * 2 - 1

        return x64, x128


# ============================================================
# 2. DataLoader
# ============================================================

def get_dataloader(root_dir, batch_size=16):
    dataset = CelebASRDataset(root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    return loader


# ============================================================
# 3. Example usage
# ============================================================

if __name__ == "__main__":
    root = "data/celeba"
    try:
        save_dir1 = f"celeb064"
        os.makedirs(save_dir1, exist_ok=True)
        save_dir2 = f"celeb128"
        os.makedirs(save_dir1, exist_ok=True)
    except IOError:
        print(e)
        exit(1)
        
        
   
           

   # loader = get_dataloader(root, batch_size=8)

    i: int =0
    
    for x64, x128 in CelebASRDataset(root):
    
        img1 = (x64.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        img2 = (x128.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        
        if i == 0:
            print("LR:", img1.shape)   # (B, 3, 64, 64)
            print("HR:", img2.shape)  # (B, 3, 128, 128)
    
        filename1 = os.path.join(save_dir1, f"img_{i+1:02d}.png")
        filename2 = os.path.join(save_dir2, f"img_{i+1:02d}.png")
        
        save_image(img1, filename1)
        save_image(img2, filename2)
        i=i+1
    