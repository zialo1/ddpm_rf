# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:03:36 2026

@author: alexh
"""

"""

1. REMOVE:
    * beta schedules
    * alphas
    * q_sample diffusion process
    * iterative sampling loop
    * epsilon prediction objective
2. REPLACE WITH:
    * linear interpolation path
    * velocity prediction
    * one-step generation
    
IMPROVEMENTS:
 1   VERY IMPORTANT FIX

Your current U-Net has a subtle issue:
nn.Conv2d(... stride=2)


2.     Pooling loses information and hurts generation quality.
3. Another important improvement

Your current upsampling:
    is weak for generative models.
 
"""

# -*- coding: utf-8 -*-
"""
============================================================
RECTIFIED FLOW (ONE STEP)
============================================================

Features:
---------
- Rectified Flow
- One-step generation
- U-Net backbone
- Sinusoidal timestep embeddings
- Residual blocks
- Conv downsampling
- ConvTranspose upsampling
- Continuous time t in [0,1]

============================================================
RECTIFIED FLOW EQUATIONS
============================================================

Interpolation path:

x_t = (1 - t)x_0 + t z

Target velocity:

v*(x_t,t) = z - x_0

Training objective:

L = || v_theta(x_t,t) - (z - x_0) ||²

One-step sampling:

x_0 = z - v_theta(z,1)

============================================================
"""

import os
import math

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image


print("running rectified flow")


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 64

BATCH_SIZE = 4

EPOCHS = 20

LR = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_DIR = "./celeb064"

OUT_DIR = "./out"

CHECKPOINT_DIR = "./checkpoints"

os.makedirs(OUT_DIR, exist_ok=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

class ImageFolderDataset(Dataset):

    def __init__(self, folder, transform=None):

        self.folder = folder

        self.transform = transform

        self.image_paths = []

        for file in os.listdir(folder):

            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):

                self.image_paths.append(
                    os.path.join(folder, file)
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        path = self.image_paths[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


# ============================================================
# SINUSOIDAL TIME EMBEDDINGS
# ============================================================

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, time):

        device = time.device

        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(
                half_dim,
                device=device
            ) * -emb
        )

        emb = time[:, None] * emb[None, :]

        emb = torch.cat(
            (
                emb.sin(),
                emb.cos()
            ),
            dim=-1
        )

        return emb


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(

            nn.SiLU(),

            nn.Linear(
                time_emb_dim,
                out_channels
            )
        )

        self.block1 = nn.Sequential(

            nn.GroupNorm(
                8,
                in_channels
            ),

            nn.SiLU(),

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            )
        )

        self.block2 = nn.Sequential(

            nn.GroupNorm(
                8,
                out_channels
            ),

            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            )
        )

        self.residual_conv = (

            nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

            if in_channels != out_channels

            else nn.Identity()
        )

    def forward(self, x, t):

        h = self.block1(x)

        time_emb = self.time_mlp(t)

        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        h = self.block2(h)

        return h + self.residual_conv(x)


# ============================================================
# U-NET
# ============================================================

class SimpleUNet(nn.Module):

    def __init__(self):
        super().__init__()

        image_channels = 3

        base_channels = 64

        time_emb_dim = 256

        # ====================================================
        # TIME EMBEDDING
        # ====================================================

        self.time_mlp = nn.Sequential(

            SinusoidalPositionEmbeddings(
                time_emb_dim
            ),

            nn.Linear(
                time_emb_dim,
                time_emb_dim
            ),

            nn.SiLU(),

            nn.Linear(
                time_emb_dim,
                time_emb_dim
            )
        )

        # ====================================================
        # INPUT
        # ====================================================

        self.init_conv = nn.Conv2d(
            image_channels,
            base_channels,
            3,
            padding=1
        )

        # ====================================================
        # DOWN
        # ====================================================

        self.down1 = ResidualBlock(
            base_channels,
            128,
            time_emb_dim
        )

        self.downsample1 = nn.Conv2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        self.down2 = ResidualBlock(
            128,
            256,
            time_emb_dim
        )

        self.downsample2 = nn.Conv2d(
            256,
            256,
            4,
            stride=2,
            padding=1
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = ResidualBlock(
            256,
            256,
            time_emb_dim
        )

        # ====================================================
        # UP
        # ====================================================

        self.upsample1 = nn.ConvTranspose2d(
            256,
            256,
            4,
            stride=2,
            padding=1
        )

        self.up1 = ResidualBlock(
            256 + 256,
            128,
            time_emb_dim
        )

        self.upsample2 = nn.ConvTranspose2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        self.up2 = ResidualBlock(
            128 + 128,
            64,
            time_emb_dim
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.final_conv = nn.Conv2d(
            64,
            image_channels,
            1
        )

    def forward(self, x, timestep):

        timestep = timestep.float()

        t = self.time_mlp(timestep)

        # ====================================================
        # INPUT
        # ====================================================

        x = self.init_conv(x)

        # ====================================================
        # DOWN 1
        # ====================================================

        x1 = self.down1(x, t)

        x = self.downsample1(x1)

        # ====================================================
        # DOWN 2
        # ====================================================

        x2 = self.down2(x, t)

        x = self.downsample2(x2)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        x = self.bottleneck(x, t)

        # ====================================================
        # UP 1
        # ====================================================

        x = self.upsample1(x)

        x = torch.cat(
            [x, x2],
            dim=1
        )

        x = self.up1(x, t)

        # ====================================================
        # UP 2
        # ====================================================

        x = self.upsample2(x)

        x = torch.cat(
            [x, x1],
            dim=1
        )

        x = self.up2(x, t)

        # ====================================================
        # OUTPUT
        # ====================================================

        return self.final_conv(x)


# ============================================================
# RECTIFIED FLOW
# ============================================================

class RectifiedFlow(nn.Module):

    def __init__(
        self,
        model,
        device="cuda"
    ):
        super().__init__()

        self.model = model

        self.device = device

    # ========================================================
    # INTERPOLATION
    #
    # x_t = (1-t)x0 + tz
    #
    # ========================================================

    def sample_xt(self, x0, t, z):

        t = t[:, None, None, None]

        xt = (
            (1.0 - t) * x0
            +
            t * z
        )

        return xt

    # ========================================================
    # TRAINING LOSS
    #
    # target velocity:
    #
    # z - x0
    #
    # ========================================================

    def p_losses(self, x0):

        batch_size = x0.shape[0]

        # continuous random time
        t = torch.rand(
            batch_size,
            device=self.device
        )

        # gaussian noise
        z = torch.randn_like(x0)

        # interpolation point
        xt = self.sample_xt(
            x0,
            t,
            z
        )

        # target velocity
        target_v = z - x0

        # predict velocity
        pred_v = self.model(
            xt,
            t
        )

        # MSE loss
        loss = F.mse_loss(
            pred_v,
            target_v
        )

        return loss

    # ========================================================
    # ONE STEP SAMPLING
    #
    # x0 = z - v_theta(z,1)
    #
    # ========================================================

    @torch.no_grad()
    def sample(self, batch_size=4):

        z = torch.randn(
            (
                batch_size,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            device=self.device
        )

        t = torch.ones(
            batch_size,
            device=self.device
        )

        v = self.model(
            z,
            t
        )

        x0 = z - v

        return x0


# ============================================================
# DATA TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# DATASET
# ============================================================

dataset = ImageFolderDataset(
    SAMPLES_DIR,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = SimpleUNet().to(DEVICE)

flow = RectifiedFlow(
    model=model,
    device=DEVICE
).to(DEVICE)

optimizer = torch.optim.AdamW(
    flow.parameters(),
    lr=LR
)


# ============================================================
# TRAINING
# ============================================================

print("Starting training...")


for epoch in range(EPOCHS):

    epoch_loss = 0.0

    i = 0

    for images in dataloader:

        images = images.to(DEVICE)

        optimizer.zero_grad()

        loss = flow.p_losses(images)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if i % 100 == 0:

            print(
                f"{i}, "
                f"loss={loss.item():.6f}"
            )

        i += 1

    avg_loss = epoch_loss / len(dataloader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f}"
    )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    if epoch + 1 == EPOCHS // 2:

        torch.save(

            flow.state_dict(),

            os.path.join(
                CHECKPOINT_DIR,
                "rf_halfway.pth"
            )
        )

        print("Saved halfway checkpoint.")

    if epoch + 1 == EPOCHS:

        torch.save(

            flow.state_dict(),

            os.path.join(
                CHECKPOINT_DIR,
                "rf_final.pth"
            )
        )

        print("Saved final checkpoint.")


# ============================================================
# GENERATE SAMPLES
# ============================================================

print("Generating samples...")

samples = flow.sample(batch_size=8)

# [-1,1] -> [0,1]
samples = (samples + 1) / 2

samples = samples.clamp(0, 1)

for i, image in enumerate(samples):

    save_path = os.path.join(
        OUT_DIR,
        f"sample_{i}.png"
    )

    save_image(
        image,
        save_path
    )

print("Finished.")


if __name__ == "__main__":
    pass