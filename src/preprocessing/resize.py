import os
import cv2

def resize_images(input_dir, output_dir="Data/processed/resized", size=(256, 256)):
    """
    Redimensiona todas las imágenes del dataset al tamaño indicado (por defecto 256x256).
    Mantiene la estructura de carpetas por especie.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n📐 Starting RESIZE...")
    print(f"Input  directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resize size: {size}\n")

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
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Redimensionar imagen
            resized = cv2.resize(img, size)

            # Guardar imagen
            out_path = os.path.join(out_species_dir, file)
            cv2.imwrite(out_path, resized)

        print(f"✔ Finished {species}")

    print("\n✨ RESIZE COMPLETED!")
    print(f"Resized images saved in: {output_dir}\n")
