# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:07:22 2026

@author: alexh
"""

"""
====================================================================
IMPACT SCORE EVALUATION FOR DDPM + RECTIFIED FLOW
====================================================================

Dieses Skript bewertet generierte Bilder aus:

    ./out/

und erzeugt einen "Impact Score" pro Bild.

Der Code ist modellagnostisch:
- funktioniert mit DDPM Outputs
- funktioniert mit Rectified Flow Outputs
- funktioniert allgemein mit generierten Bildern

====================================================================
WAS IST DER IMPACT SCORE?
====================================================================

Der Impact Score kombiniert:

1. Schärfe
2. Kontrast
3. Entropie
4. Ästhetische Struktur
5. Diversity-artige Bildkomplexität

Die Idee:
----------
Ein gutes generatives Modell produziert Bilder mit:

- klaren Kanten
- sinnvollen Strukturen
- hohem Informationsgehalt
- nicht nur glattem Rauschen

====================================================================
WARUM DAS FÜR DDPM vs RECTIFIED FLOW NÜTZLICH IST
====================================================================

DDPM:
------
- oft glatter
- manchmal weichere Bilder

Rectified Flow:
---------------
- oft schärfer
- kontrastreicher
- aggressivere Struktur

Der Score misst diese Unterschiede numerisch.

====================================================================
OUTPUT
====================================================================

Das Skript erzeugt:

1. Konsole:
    - Score pro Bild
    - Durchschnittsscore

2. CSV Datei:
    ./out/impact_scores.csv

====================================================================
INSTALLATION
====================================================================

pip install torch torchvision pillow numpy pandas opencv-python scipy

====================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from scipy.stats import entropy


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = "./out/" # '/'

CSV_OUTPUT = "./out/impact_scores.csv"


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    """
    Lädt Bild als:
    - RGB
    - numpy array
    """

    image = Image.open(path).convert("RGB")

    image = np.array(image)

    return image


# ============================================================
# SHARPNESS SCORE
# ============================================================
#
# Misst:
# wie viele starke Kanten existieren.
#
# Verwendet:
# Laplacian Variance
#
# Hoher Wert:
# -> scharfes Bild
#
# Niedriger Wert:
# -> verschwommen
# ============================================================

def sharpness_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    score = laplacian.var()

    return float(score)


# ============================================================
# CONTRAST SCORE
# ============================================================
#
# Standardabweichung der Pixelintensitäten.
#
# Hoher Kontrast:
# -> größere visuelle Wirkung
# ============================================================

def contrast_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    return float(gray.std())


# ============================================================
# ENTROPY SCORE
# ============================================================
#
# Misst:
# Informationsgehalt
#
# Höhere Entropie:
# -> komplexere Bilder
# ============================================================

def entropy_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist.ravel()

    hist /= hist.sum()

    return float(entropy(hist))


# ============================================================
# EDGE DENSITY
# ============================================================
#
# Misst:
# Wie viele Kanten im Bild existieren.
#
# Rectified Flow erzeugt oft:
# - mehr Edge Density
#
# DDPM erzeugt oft:
# - weichere Übergänge
# ============================================================

def edge_density_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        threshold1=100,
        threshold2=200
    )

    density = edges.mean()

    return float(density)


# ============================================================
# COLOR VARIANCE
# ============================================================
#
# Misst:
# Farbdiversität
# ============================================================

def color_variance_score(image):

    return float(np.var(image))


# ============================================================
# IMPACT SCORE
# ============================================================
#
# Gewichtete Kombination aller Scores.
#
# Diese Gewichte kannst du später tunen.
# ============================================================

def impact_score(image):

    sharpness = sharpness_score(image)

    contrast = contrast_score(image)

    entropy_val = entropy_score(image)

    edge_density = edge_density_score(image)

    color_var = color_variance_score(image)

    # ========================================================
    # NORMALISIERTE KOMBINATION
    # ========================================================

    score = (

        0.30 * sharpness +

        0.20 * contrast +

        0.20 * entropy_val * 100 +

        0.20 * edge_density +

        0.10 * color_var / 100
    )

    return {

        "sharpness": sharpness,

        "contrast": contrast,

        "entropy": entropy_val,

        "edge_density": edge_density,

        "color_variance": color_var,

        "impact_score": score
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_directory(directory):

    results = []

    image_files = []

    for file in os.listdir(directory):

        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            image_files.append(file)

    image_files.sort()

    print("=" * 60)
    print("EVALUATING IMAGES")
    print("=" * 60)

    for file in image_files:

        path = os.path.join(directory, file)

        image = load_image(path)

        scores = impact_score(image)

        row = {
            "image": path,
            **scores
        }

        results.append(row)

        print(f"\nImage: {file}")

        print(
            f"Impact Score: "
            f"{scores['impact_score']:.2f}"
        )

        print(
            f"Sharpness: "
            f"{scores['sharpness']:.2f}"
        )

        print(
            f"Contrast: "
            f"{scores['contrast']:.2f}"
        )

        print(
            f"Entropy: "
            f"{scores['entropy']:.2f}"
        )

        print(
            f"Edge Density: "
            f"{scores['edge_density']:.2f}"
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(results)

    # ========================================================
    # SAVE CSV
    # ========================================================

    df.to_csv(
        CSV_OUTPUT,
        index=False
    )

    # ========================================================
    # GLOBAL STATS
    # ========================================================

    print("\n" + "=" * 60)

    print("SUMMARY")

    print("=" * 60)

    print(
        f"Average Impact Score: "
        f"{df['impact_score'].mean():.2f}"
    )

    print(
        f"Best Image: "
        f"{df.iloc[df['impact_score'].idxmax()]['image']}"
    )

    print(
        f"Worst Image: "
        f"{df.iloc[df['impact_score'].idxmin()]['image']}"
    )

    print(
        f"\nSaved CSV to:\n{CSV_OUTPUT}"
    )

    return df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    for root, dirs, files in os.walk(OUT_DIR):
        for k in dirs:
            if k[0] == '.':
                next
            workdir =  root +  k
            print("working on dir (saving report there)=",workdir)
            evaluate_directory(workdir)
        break
