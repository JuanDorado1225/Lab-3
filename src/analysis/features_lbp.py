import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

RADIUS = 1
POINTS = 8 * RADIUS
METHOD = "uniform"

def extract_lbp_features(input_dir, output_csv="Data/features/features_lbp.csv"):
    data_list = []

    print("\n🔬 Starting LBP FEATURE EXTRACTION...\n")

    for species in os.listdir(input_dir):
        species_path = os.path.join(input_dir, species)
        if not os.path.isdir(species_path):
            continue

        for file in os.listdir(species_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(species_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            lbp = local_binary_pattern(img, POINTS, RADIUS, METHOD)

            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

            uniform_ratio = hist[:-1].sum()
            entropy = shannon_entropy(hist + 1e-12)
            dom_bin_ratio = hist.max() / hist.sum()

            data_list.append({
                "filename": file,
                "species": species,
                "lbp_uniform_ratio": uniform_ratio,
                "lbp_entropy": entropy,
                "lbp_dom_bin_ratio": dom_bin_ratio
            })

    df = pd.DataFrame(data_list)

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"✔ LBP features saved in: {output_csv}\n")
    return df
