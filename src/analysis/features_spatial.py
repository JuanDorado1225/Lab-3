import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy

def extract_spatial_features(input_dir, output_csv="Data/features/features_spatial.csv"):
    data_list = []

    print("\n📊 Starting SPATIAL FEATURE EXTRACTION...")
    print(f"Input directory: {input_dir}\n")

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

            # Features
            mean_intensity = np.mean(img)
            contrast = np.std(img)
            entropy = shannon_entropy(img)
            edges = cv2.Canny(img, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            data_list.append({
                "filename": file,
                "species": species,
                "mean_intensity": mean_intensity,
                "contrast": contrast,
                "entropy": entropy,
                "edge_density": edge_density
            })

    df = pd.DataFrame(data_list)

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"✔ Spatial features saved in: {output_csv}\n")
    return df
