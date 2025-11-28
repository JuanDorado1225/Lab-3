import os
import cv2
import numpy as np

def is_valid_image(path, min_width=200, min_height=200):
    """Check if an image is readable, has minimum size, and is not fully black/white."""
    img = cv2.imread(path)

    # 1. Archivo dañado o ilegible
    if img is None:
        return False, "Unreadable/Corrupted"

    h, w = img.shape[:2]

    # 2. Tamaño mínimo
    if w < min_width or h < min_height:
        return False, f"Too small ({w}x{h})"

    # 3. Imagen completamente negra o blanca
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 2:  # sin variación → podría estar negra o blanca
        return False, "Almost blank (low variance)"

    return True, "OK"


def clean_dataset(input_dir, output_dir="Data/processed/clean"):
    """
    Cleans dataset by filtering corrupted, tiny, or blank images.
    Keeps same folder structure (one folder per species).
    """
    os.makedirs(output_dir, exist_ok=True)

    rejected_log = []  # para registrar imágenes descartadas

    print("\n🔍 Starting DATA CLEANING...")
    print(f"Input  directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")

    # Recorrer especies
    for species in os.listdir(input_dir):
        species_path = os.path.join(input_dir, species)

        if not os.path.isdir(species_path):
            continue

        print(f"Processing species: {species}")

        # Crear carpeta destino
        out_species_dir = os.path.join(output_dir, species)
        os.makedirs(out_species_dir, exist_ok=True)

        # Recorrer imágenes
        for file in os.listdir(species_path):
            path = os.path.join(species_path, file)

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            valid, reason = is_valid_image(path)

            if valid:
                # Copiar imagen válida al destino
                img = cv2.imread(path)
                cv2.imwrite(os.path.join(out_species_dir, file), img)
            else:
                # Registrar imagen descartada
                rejected_log.append((path, reason))

        print(f"✔ Finished {species}")

    # Guardar log de imágenes rechazadas con codificación UTF-8
    log_path = os.path.join(output_dir, "rejected_images.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for path, reason in rejected_log:
            f.write(f"{path} -> {reason}\n")

    print("\n✨ CLEANING COMPLETED")
    print(f"Valid images saved in: {output_dir}")
    print(f"Rejected images logged in: {log_path}\n")
