import os
import cv2
import numpy as np

def enhance_images(input_dir, output_dir="Data/processed/enhanced", 
                   apply_equalization=True, apply_clahe=True, apply_normalization=True):
    """
    Aplica mejoras de contraste:
    - Histogram Equalization (global)
    - CLAHE (local adaptative)
    - Normalization 0–255
    Mantiene estructura por especie.
    Solo funciona con imágenes en escala de grises.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    print("\n⚡ Starting ENHANCEMENT...")
    print(f"Input  directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Options: Equalization={apply_equalization}, CLAHE={apply_clahe}, Normalization={apply_normalization}\n")

    # Crear objeto CLAHE (solo si está activado)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if apply_clahe else None

    for species in os.listdir(input_dir):
        species_path = os.path.join(input_dir, species)
        
        if not os.path.isdir(species_path):
            continue

        print(f"Processing species: {species}")

        out_species_dir = os.path.join(output_dir, species)
        os.makedirs(out_species_dir, exist_ok=True)

        for file in os.listdir(species_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(species_path, file)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                continue

            enhanced = gray.copy()

            # --- Histogram Equalization ---
            if apply_equalization:
                enhanced = cv2.equalizeHist(enhanced)

            # --- CLAHE ---
            if apply_clahe:
                enhanced = clahe.apply(enhanced)

            # --- Normalization ---
            if apply_normalization:
                enhanced = cv2.normalize(enhanced, None, alpha=0, beta=255, 
                                         norm_type=cv2.NORM_MINMAX)

            # Guardar imagen mejorada
            out_path = os.path.join(out_species_dir, file)
            cv2.imwrite(out_path, enhanced)

        print(f"✔ Finished {species}")

    print("\n✨ ENHANCEMENT COMPLETED!")
    print(f"Enhanced images saved in: {output_dir}\n")
