#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:01:31 2026

@author: alex

* RF produces samples
* discriminator ranks them
* SR learns from hardest cases

✔ one-step RF generator
✔ discriminator selecting difficult samples
✔ SR-DDPM trained on hard distribution gaps

We stabilize the discriminator using:

✔ 1. Label smoothing

* real = 0.9 instead of 1.0

✔ 2. Instance noise

* add small noise to inputs

✔ 3. Gradient penalty (lightweight R1)

* prevents exploding gradients

✔ 4. Balanced training

* update disc less often (e.g. every 2 steps)

✔ 5. Temperature-controlled sampling

* avoids collapsing to a few RF samples

knoobs

temperature = 2.0 – 5.0
sigma = 0.02 – 0.1
10.0 (good default)


files in folder hardrf have form 
rf_t3C00_s012_045_233_101_000123.png

temperature, s012_045_233_101 discriminator scores (scaled, abs, 3-digit), 
stepcounter

tip
Watch the saved images:

* if they look too similar → temperature too high
* if they look random → discriminator too weak
* if they look too clean → you’re not sampling “hard” enough


"""


import os,sys,math
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import glob

import torchvision.utils as vutils

device = "cuda" if torch.cuda.is_available() else "cpu"



# ============================================================
# 1. Dataset
# ============================================================

class CelebAPairedDataset(Dataset):
    def __init__(self, dir_64, dir_128):
        self.dir_64 = dir_64
        self.dir_128 = dir_128

        files_64 = set(os.listdir(dir_64))
        files_128 = set(os.listdir(dir_128))
        self.files = sorted(list(files_64 & files_128))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]

        img64 = Image.open(os.path.join(self.dir_64, f)).convert("RGB")
        img128 = Image.open(os.path.join(self.dir_128, f)).convert("RGB")

        x64 = self.to_tensor(img64) * 2 - 1
        x128 = self.to_tensor(img128) * 2 - 1

        return x64, x128


class DataModule:
    def __init__(self, dir_64, dir_128, batch_size=16):
        self.loader = DataLoader(
            CelebAPairedDataset(dir_64, dir_128),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )


# ============================================================
# 2. One-step Rectified Flow
# ============================================================

class OneStepRF:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def make_onestep(self, x):
        self.model.eval()
        t0 = torch.zeros(x.size(0), device=self.device)
        v = self.model(x, t0)
        return x + v

    def sample(self, batch_size):
        x = torch.randn(batch_size, 3, 64, 64, device=self.device)
        x = self.make_onestep(x)
        return x.clamp(-1, 1)


# ============================================================
# 3. Stable Discriminator
# ============================================================

class RFDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5. Discriminator Trainer
# ============================================================

class DiscriminatorTrainer:
    def __init__(self, disc, rf, device):
        self.disc = disc
        self.rf = rf
        self.device = device

    def add_noise(self, x, sigma=0.05):
        return x + sigma * torch.randn_like(x)

    def r1_penalty(self, pred, x):
        grad = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=x,
            create_graph=True
        )[0]
        return grad.pow(2).reshape(x.size(0), -1).sum(1).mean()

    def train_step(self, x_real, optimizer, step):
        B = x_real.size(0)

        real_labels = torch.full((B, 1), 0.9, device=self.device)
        fake_labels = torch.zeros(B, 1, device=self.device)

        with torch.no_grad():
            x_fake = self.rf.sample(B)

        x_real = self.add_noise(x_real)
        x_fake = self.add_noise(x_fake)

        x_real.requires_grad_(True)

        pred_real = self.disc(x_real)
        pred_fake = self.disc(x_fake.detach())

        loss = (
            F.binary_cross_entropy_with_logits(pred_real, real_labels) +
            F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
        )

        if step % 4 == 0:
            loss += 10.0 * self.r1_penalty(pred_real, x_real)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


# ============================================================
# 6. SR Trainer
# ============================================================

class SRTrainer:
    def __init__(self, model, sampler, device):
        self.model = model
        self.sampler = sampler
        self.device = device

    def train_step(self, x64_real, x128, optimizer):
        B = x64_real.size(0)
        half = B // 2

        x64_rf = self.sampler.sample(half)

        x64 = torch.cat([x64_real[:half], x64_rf], dim=0)

        perm = torch.randperm(B)
        x64 = x64[perm]
        x128 = x128[perm]

        t = torch.rand(B, device=self.device)
        noise = torch.randn_like(x128)

        x_t = (1 - t[:, None, None, None]) * x128 + \
              t[:, None, None, None] * noise

        pred = self.model(x_t, x64, t)

        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


# ============================================================
# 7. Main Trainer
# ============================================================

class Trainer:
    def __init__(self, model, rf_model, device, loader):
        self.device = device

        self.rf = OneStepRF(rf_model, device)
        self.disc = RFDiscriminator().to(device)

        self.sampler = HardRFSampler(self.rf, self.disc, device)
        self.sr_trainer = SRTrainer(model, self.sampler, device)
        self.disc_trainer = DiscriminatorTrainer(self.disc, self.rf, device)

        self.loader = loader

        self.opt_sr = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=1e-4)

    def train(self, epochs=10):
        step = 0

        for epoch in range(epochs):
            for x64, x128 in self.loader:
                x64 = x64.to(self.device)
                x128 = x128.to(self.device)

                # train discriminator (less frequent)
                if step % 2 == 0:
                    self.disc_trainer.train_step(x64, self.opt_d, step)

                # train SR
                self.sr_trainer.train_step(x64, x128, self.opt_sr)

                step += 1
                
class HardRFSampler:
    def __init__(self, rf, disc, device, save_dir="hard_rf", temperature=3.0):
        self.rf = rf
        self.disc = disc
        self.device = device
        self.temperature = temperature

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.counter = 0

    def sample(self, batch_size):
        with torch.no_grad():
            x_rf = self.rf.sample(batch_size)

            scores = self.disc(x_rf).squeeze()   # shape: (B,)
            hardness = -scores

            prob = torch.softmax(hardness * self.temperature, dim=0)
            idx = torch.multinomial(prob, batch_size)

            x_hard = x_rf[idx]
            selected_scores = scores[idx]

            self._save_samples(x_hard, selected_scores)

            return x_hard

    def _format_float(self, value):
        """
        Convert float like 3.0 → '3C00'
        """
        s = f"{value:.2f}"   # '3.00'
        return s.replace(".", "C")

    def _format_score(self, score):
        """
        Convert discriminator score to 3-digit integer string.
        Example: -0.23 → 023
        """
        score_abs = torch.abs(score)
        score_int = int((score_abs * 100).clamp(0, 999).item())
        return f"{score_int:03d}"

    def _save_samples(self, x, scores):
        x_vis = (x + 1) / 2  # [-1,1] → [0,1]

        temp_str = self._format_float(self.temperature)

        # build filename with multiple scores (first 4 shown)
        score_strs = [self._format_score(s) for s in scores[:4]]
        score_block = "_".join(score_strs)

        fname = f"rf_t{temp_str}_s{score_block}_{self.counter:06d}.png"
        path = os.path.join(self.save_dir, fname)

        vutils.save_image(x_vis, path, nrow=4)

        self.counter += R1
        
# --- RF MODEL CODE

# =========================================
# Rectified Flow + DDPM++ U-Net on CelebA-64
# =========================================


# =========================================
# Time Embedding
# =========================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# =========================================
# Residual Block
# =========================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time = nn.Linear(time_dim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# =========================================
# Attention
# =========================================
class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = self.norm(x)

        q, k, v = self.qkv(h_in).chunk(3, dim=1)

        q = q.reshape(b, c, -1)
        k = k.reshape(b, c, -1)
        v = v.reshape(b, c, -1)

        attn = torch.softmax((q.transpose(1, 2) @ k) / math.sqrt(c), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2)

        out = out.reshape(b, c, h, w)
        return x + self.proj(out)

# -----------------------------
# 1. Simple UNet (minimal)
# -----------------------------

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base),
            nn.ReLU(),
            nn.Linear(base, base)
        )

        self.conv1 = nn.Conv2d(in_channels, base, 3, padding=1)
        self.conv2 = nn.Conv2d(base, base, 3, padding=1)
        self.conv3 = nn.Conv2d(base, in_channels, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, t):
        # t: (B,) → (B,1)
        t = t.view(-1, 1)
        t_embed = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(x))
        h = h + t_embed
        h = self.act(self.conv2(h))
        out = self.conv3(h)

        return out
    
# =========================================
# DDPM++ U-Net
# =========================================
class UNetDDPM(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()

        time_dim = base * 4

        self.time_embed = nn.Sequential(
            TimeEmbedding(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down
        self.down1 = ResBlock(base, base * 2, time_dim)      # 64 → 128
        self.down2 = ResBlock(base * 2, base * 4, time_dim)  # 128 → 256

        # Middle
        self.mid = ResBlock(base * 4, base * 4, time_dim)    # 256 → 256
        self.attn_mid = Attention(base * 4)

        # Up (FIXED CHANNELS)
        self.up1 = ResBlock(base * 8, base * 2, time_dim)    # (256+256)=512 → 128
        self.up2 = ResBlock(base * 4, base, time_dim)        # (128+128)=256 → 64

        self.out = nn.Conv2d(base, in_ch, 1)

        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, t):
        t = self.time_embed(t)

        x = self.init(x)          # 64

        x1 = self.down1(x, t)     # 128
        x1d = self.downsample(x1)

        x2 = self.down2(x1d, t)   # 256
        x2d = self.downsample(x2)

        h = self.mid(x2d, t)
        h = self.attn_mid(h)

        # Up 1
        h = self.upsample(h)              # 256
        h = torch.cat([h, x2], dim=1)     # 256+256=512
        h = self.up1(h, t)                # → 128

        # Up 2
        h = self.upsample(h)              # 128
        h = torch.cat([h, x1], dim=1)     # 128+128=256
        h = self.up2(h, t)                # → 64

        return self.out(h)

# =========================================
# Rectified Flow
# =========================================
class RectifiedFlow:
    def __init__(self, model):
        self.model = model.to(device)
        self.opt = optim.Adam(model.parameters(), lr=1e-4)

    def train_step(self, x1):
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)

        t = torch.rand(x1.size(0), device=device)

        t_ = t[:, None, None, None]

        xt = (1 - t_) * x0 + t_ * x1
        v_target = x1 - x0

        v_pred = self.model(xt, t)

        loss = ((v_pred - v_target) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, n=16, steps=50):
        x = torch.randn(n, 3, 64, 64).to(device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((n,), i / steps, device=device)
            x = x + dt * self.model(x, t)

        return x
    
    @torch.no_grad()
    def generate_pairs(self, x0, steps=50):
        """
        Takes noise x0 → generates x1_hat using current model
        """
        self.model.eval()
    
        x = x0.clone()
        dt = 1.0 / steps
    
        for i in range(steps):
            t = torch.full((x.size(0),), i / steps, device=x.device)
            v = self.model(x, t)
            x = x + dt * v
    
        x1_hat = x
        return x0, x1_hat

    def makeonestep(self,x):
        self.model.eval()
        t0 = torch.zeros(16, device=device)         
        v = self.model(x,t0)
        return x+v



# --- END RF MODEL CODE
def main(amodel):
    dir64 = "celeb064"
    dir128 = "celeb128"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------
    loader = DataModule(dir64, dir128).loader

    # --------------------------------------------------------
    # 2. Rectified Flow model
    # --------------------------------------------------------
    basemodel = SimpleUNet().to(device)
    rf = RectifiedFlow(basemodel)

    ckpt = torch.load(amodel, map_location=device)
    rf.model.load_state_dict(ckpt)

    rf.model.eval()
    print(f"[OK] RF model loaded: {amodel}")

    # --------------------------------------------------------
    # 3. Wrap RF properly (IMPORTANT FIX)
    # --------------------------------------------------------
    rf_wrapper = OneStepRF(rf.model, device)

    # --------------------------------------------------------
    # 4. Discriminator
    # --------------------------------------------------------
    disc = RFDiscriminator().to(device)

    # --------------------------------------------------------
    # 5. Trainer
    # --------------------------------------------------------
    trainer = Trainer(
        model=RFDiscriminator().to(device),  # your SR DDPM model
        rf_model=rf.model,
        device=device,
        loader=loader
    )

    # IMPORTANT: inject correct RF wrapper into trainer
    trainer.rf = rf_wrapper
    trainer.disc = disc

    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------
    trainer.train()
    

if __name__ == '__main__':
    main("src/rf_ddpmpp_celeba64_0R1.pth")
    
    
    
        
        
