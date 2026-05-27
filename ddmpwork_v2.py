# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:03:36 2026

@author: alexh
"""

"""
====================================================================
ORIGINALES DDPM (DENOISING DIFFUSION PROBABILISTIC MODEL)
====================================================================

Paper:
" Denoising Diffusion Probabilistic Models "
Ho et al. 2020

====================================================================
GRUNDIDEE
====================================================================

Ein Bild wird schrittweise verrauscht:

x0 -> x1 -> x2 -> ... -> xT

Nach vielen Schritten bleibt nur Gaußsches Rauschen übrig.

Das Modell lernt anschließend den Umkehrprozess:

xT -> xT-1 -> ... -> x0

Dadurch kann das Netzwerk neue Bilder generieren.

====================================================================
WICHTIGE FORMELN AUS DEM PAPER
====================================================================

------------------------------------------------------------
1. FORWARD DIFFUSION PROCESS
------------------------------------------------------------

q(x_t | x_{t-1}) =
    N(
        sqrt(1 - beta_t) * x_{t-1},
        beta_t * I
    )

Bedeutung:
-----------
Bei jedem Schritt wird etwas Gaußrauschen hinzugefügt.

beta_t:
    Stärke des hinzugefügten Rauschens.

====================================================================

------------------------------------------------------------
2. GESCHLOSSENE FORM DES FORWARD PROCESS
------------------------------------------------------------

q(x_t | x_0) =
    N(
        sqrt(alpha_bar_t) * x_0,
        (1 - alpha_bar_t) * I
    )

mit:

alpha_t = 1 - beta_t

alpha_bar_t =
    Produkt(alpha_i) von i=1 bis t

====================================================================

------------------------------------------------------------
3. REPARAMETRISIERUNG
------------------------------------------------------------

x_t =
    sqrt(alpha_bar_t) * x_0
    +
    sqrt(1 - alpha_bar_t) * epsilon

epsilon ~ N(0, I)

Bedeutung:
-----------
Ein ver-
- Zufallsrauschen

berechnet werden.

====================================================================

------------------------------------------------------------
4. TRAININGSZIEL
------------------------------------------------------------

L_simple =
    E[
        || epsilon - epsilon_theta(x_t, t) ||²
    ]

Das Netzwerk soll das hinzugefügte Rauschen vorhersagen.

epsilon:
    echtes Rauschen

epsilon_theta:
    vorhergesagtes Rauschen

====================================================================

------------------------------------------------------------
5. REVERSE PROCESS
------------------------------------------------------------

p_theta(x_{t-1} | x_t) =
    N(
        mu_theta(x_t, t),
        Sigma_theta(x_t, t)
    )

Das Modell approximiert den Umkehrprozess.

====================================================================

------------------------------------------------------------
6. SAMPLING UPDATE
------------------------------------------------------------

x_{t-1} =
    1/sqrt(alpha_t)
    *
    (
        x_t
        -
        ((1 - alpha_t) /
        sqrt(1 - alpha_bar_t))
        *
        epsilon_theta(x_t, t)
    )
    +
    sigma_t * z

z ~ N(0, I)

Bedeutung:
-----------
Das Modell entfernt iterativ Rauschen.

====================================================================
IMPLEMENTATION
====================================================================

Dieses Skript implementiert:

- originales DDPM
- linearen beta schedule
- epsilon prediction objective
- U-Net backbone
- vollständiges Sampling
- Checkpoints
- Bildspeicherung

====================================================================
"""

import os
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

print("running new file")
# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64

BATCH_SIZE = 1

EPOCHS = 20

TIMESTEPS = 1000

LR = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES_DIR = "./celeb064"

OUT_DIR = "./out"

CHECKPOINT_DIR = "./checkpoints"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# CUSTOM DATASET
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
#
# Das Modell muss wissen:
# "In welchem Diffusionsschritt befinden wir uns?"
#
# Dafür wird der Zeitschritt t eingebettet.
#
# Ähnlich zu Transformer Position Embeddings.
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
            torch.arange(half_dim, device=device) * -emb
        )

        emb = time[:, None] * emb[None, :]

        emb = torch.cat(
            (emb.sin(), emb.cos()),
            dim=-1
        )

        return emb


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.time_mlp = nn.Linear(
            time_emb_dim,
            out_channels
        )

        self.block1 = nn.Sequential(

            nn.GroupNorm(8, in_channels),

            nn.SiLU(),

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            )
        )

        self.block2 = nn.Sequential(

            nn.GroupNorm(8, out_channels),

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
# SIMPLE U-NET
# ============================================================

class SimpleUNet(nn.Module):

    def __init__(self):
        super().__init__()

        image_channels = 3

        down_channels = (64, 128, 256)

        up_channels = (256, 128, 64)

        time_emb_dim = 256

        self.time_mlp = nn.Sequential(

            SinusoidalPositionEmbeddings(time_emb_dim),

            nn.Linear(
                time_emb_dim,
                time_emb_dim
            ),

            nn.SiLU()
        )

        self.conv0 = nn.Conv2d(
            image_channels,
            down_channels[0],
            3,
            padding=1
        )

        self.downs = nn.ModuleList([

            ResidualBlock(
                down_channels[0],
                down_channels[1],
                time_emb_dim
            ),

            ResidualBlock(
                down_channels[1],
                down_channels[2],
                time_emb_dim
            )
        ])

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(
            down_channels[2],
            down_channels[2],
            time_emb_dim
        )

        self.ups = nn.ModuleList([
        
            ResidualBlock(
                256 + 256,
                128,
                time_emb_dim
            ),
        
            ResidualBlock(
                128 + 128,
                64,
                time_emb_dim
            )
        ])


        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        self.output = nn.Conv2d(
            up_channels[-1],
            image_channels,
            1
        )

    def forward(self, x, timestep):

        t = self.time_mlp(timestep)

        x = self.conv0(x)

        residual_inputs = []

        # ====================================================
        # DOWN SAMPLING
        # ====================================================

        for down in self.downs:

            x = down(x, t)

            residual_inputs.append(x)

            x = self.pool(x)

        # ====================================================
        # BOTTLENECK
        # ====================================================

        x = self.bottleneck(x, t)

        # ====================================================
        # UP SAMPLING
        # ====================================================

        for up in self.ups:

            residual_x = residual_inputs.pop()

            x = self.upsample(x)

            if x.shape[-1] != residual_x.shape[-1]:

                x = F.interpolate(
                    x,
                    size=residual_x.shape[-2:]
                )

            x = torch.cat(
                [x, residual_x],
                dim=1
            )

            x = up(x, t)

        return self.output(x)


# ============================================================
# DDPM
# ============================================================

class DDPM(nn.Module):

    def __init__(
        self,
        model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda"
    ):
        super().__init__()

        self.model = model

        self.timesteps = timesteps

        self.device = device

        # ====================================================
        # LINEAR BETA SCHEDULE
        # ====================================================
        #
        # beta_t bestimmt:
        # wie viel Rauschen pro Schritt hinzugefügt wird.
        #
        # Original DDPM:
        # linear increasing schedule
        # ====================================================

        betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps
        ).to(device)

        # alpha_t = 1 - beta_t
        alphas = 1. - betas

        # alpha_bar_t = Produkt(alpha_i)
        alphas_cumprod = torch.cumprod(
            alphas,
            dim=0
        )

        self.register_buffer("betas", betas)

        self.register_buffer("alphas", alphas)

        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod
        )

        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod)
        )

        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1 - alphas_cumprod)
        )

    # ========================================================
    # FORWARD DIFFUSION
    # ========================================================
    #
    # PAPER:
    #
    # x_t =
    #   sqrt(alpha_bar_t) * x_0
    #   +
    #   sqrt(1 - alpha_bar_t) * epsilon
    #
    # epsilon ~ N(0, I)
    #
    # ========================================================

    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = (
            self.sqrt_alphas_cumprod[t]
            [:, None, None, None]
        )

        sqrt_one_minus_alphas_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod[t]
            [:, None, None, None]
        )

        return (

            sqrt_alphas_cumprod_t * x_start +

            sqrt_one_minus_alphas_cumprod_t * noise
        )

    # ========================================================
    # TRAINING OBJECTIVE
    # ========================================================
    #
    # PAPER:
    #
    # L_simple =
    #   E[
    #       || epsilon -
    #          epsilon_theta(x_t, t)
    #       ||²
    #   ]
    #
    # Das Modell lernt:
    # "Welches Rauschen wurde hinzugefügt?"
    #
    # ========================================================

    def p_losses(self, x_start, t):

        # Echtes Rauschen
        noise = torch.randn_like(x_start)

        # Verrauschtes Bild erzeugen
        x_noisy = self.q_sample(
            x_start,
            t,
            noise
        )

        # Modell sagt Rauschen voraus
        predicted_noise = self.model(
            x_noisy,
            t.float()
        )

        # MSE zwischen:
        # echtem und vorhergesagtem Rauschen
        loss = F.mse_loss(
            predicted_noise,
            noise
        )

        return loss

    # ========================================================
    # REVERSE SAMPLING
    # ========================================================
    #
    # PAPER:
    #
    # x_{t-1} =
    #
    # 1/sqrt(alpha_t)
    # *
    # (
    #   x_t
    #   -
    #   ((1 - alpha_t)
    #    / sqrt(1 - alpha_bar_t))
    #   *
    #   epsilon_theta(x_t, t)
    # )
    # +
    # sigma_t * z
    #
    # ========================================================

    @torch.no_grad()
    def sample(self, batch_size=4):

        # Starte mit reinem Gaußrauschen
        x = torch.randn(
            (
                batch_size,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            device=self.device
        )

        # Rückwärts durch alle Schritte
        for t in reversed(range(self.timesteps)):

            t_batch = torch.full(
                (batch_size,),
                t,
                device=self.device,
                dtype=torch.long
            )

            betas_t = self.betas[t_batch][:, None, None, None]

            sqrt_one_minus_alphas_cumprod_t = (
                self.sqrt_one_minus_alphas_cumprod[t_batch]
                [:, None, None, None]
            )

            sqrt_recip_alphas_t = (
                1.0 / torch.sqrt(self.alphas[t_batch])
            )[:, None, None, None]

            # Vorhergesagtes Rauschen
            predicted_noise = self.model(
                x,
                t_batch.float()
            )

            # Mittelwert der Reverse Distribution
            model_mean = sqrt_recip_alphas_t * (

                x -

                betas_t * predicted_noise /

                sqrt_one_minus_alphas_cumprod_t
            )

            # Zusätzliche Zufallskomponente
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            posterior_variance = betas_t

            x = model_mean + (
                torch.sqrt(posterior_variance)
                * noise
            )

        return x


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    # [0,1] -> [-1,1]
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

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
# MODEL INITIALIZATION
# ============================================================

model = SimpleUNet().to(DEVICE)

diffusion = DDPM(
    model=model,
    timesteps=TIMESTEPS,
    device=DEVICE
).to(DEVICE)

optimizer = torch.optim.Adam(
    diffusion.parameters(),
    lr=LR
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("Starting training...")

for epoch in range(EPOCHS):


    epoch_loss = 0.0

    i=0
    
    for images in dataloader:

        images = images.to(DEVICE)

        optimizer.zero_grad()

        # Zufällige Diffusionsschritte
        t = torch.randint(
            0,
            diffusion.timesteps,
            (images.shape[0],),
            device=DEVICE
        ).long()


        # DDPM Loss
        loss = diffusion.p_losses(images, t)
      
        # Backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        if i % 100 == 0:
            
            print(f"{i}, {loss}, {epoch_loss}...")
            
        i=i+1
       
    avg_loss = epoch_loss / len(dataloader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f}"
    )

    # ========================================================
    # SAVE HALF CHECKPOINT
    # ========================================================

    if epoch + 1 == EPOCHS // 2:

        torch.save(

            diffusion.state_dict(),

            os.path.join(
                CHECKPOINT_DIR,
                "ddpm_halfway.pth"
            )
        )

        print("Saved halfway checkpoint.")

    # ========================================================
    # SAVE FINAL CHECKPOINT
    # ========================================================

    if epoch + 1 == EPOCHS:

        torch.save(

            diffusion.state_dict(),

            os.path.join(
                CHECKPOINT_DIR,
                "ddpm_final.pth"
            )
        )

        print("Saved final checkpoint.")


# ============================================================
# GENERATE SAMPLES
# ============================================================

print("Generating samples...")

samples = diffusion.sample(batch_size=8)

# [-1,1] -> [0,1]
samples = (samples + 1) / 2

for i, image in enumerate(samples):

    save_path = os.path.join(
        OUT_DIR,
        f"sample_{i}.png"
    )

    save_image(image, save_path)

print("Finished.")

if __name__ == "__main__":
    pass
