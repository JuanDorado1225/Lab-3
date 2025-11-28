import os
import cv2
import numpy as np
import pandas as pd

def extract_frequency_features(input_dir, output_csv="Data/features/features_frequency.csv"):
    data_list = []

    print("\n📡 Starting FREQUENCY FEATURE EXTRACTION...\n")

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

            # FFT
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)

            spectral_energy = np.sum(magnitude ** 2)

            h, w = img.shape
            cy, cx = h // 2, w // 2
            radius = min(cx, cy) // 4

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            low_mask = dist <= radius
            high_mask = dist > radius

            low_freq_energy = np.sum(magnitude[low_mask] ** 2)
            high_freq_energy = np.sum(magnitude[high_mask] ** 2)
            high_low_ratio = high_freq_energy / (low_freq_energy + 1e-8)

            # Dominant frequency
            flat = magnitude.flatten()
            idx_max = np.argmax(flat)
            y_peak, x_peak = np.unravel_index(idx_max, magnitude.shape)
            dist_peak = np.sqrt((x_peak - cx)**2 + (y_peak - cy)**2)
            dominant_freq = dist_peak / np.sqrt(cx**2 + cy**2)

            data_list.append({
                "filename": file,
                "species": species,
                "spectral_energy": spectral_energy,
                "low_freq_energy": low_freq_energy,
                "high_freq_energy": high_freq_energy,
                "high_low_ratio": high_low_ratio,
                "dominant_frequency": dominant_freq
            })

    df = pd.DataFrame(data_list)

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"✔ Frequency features saved in: {output_csv}\n")
    return df
