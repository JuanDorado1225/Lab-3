import os
import cv2

def convert_to_grayscale(input_dir, output_dir="Data/processed/grayscale"):
    """
    Convierte todas las imágenes del dataset a escala de grises.
    Mantiene la estructura de carpetas por especie.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n🖤 Starting GRAYSCALE CONVERSION...")
    print(f"Input  directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")

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

            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Guardar imagen
            out_path = os.path.join(out_species_dir, file)
            cv2.imwrite(out_path, gray)

        print(f"✔ Finished {species}")

    print("\n✨ GRAYSCALE COMPLETED!")
    print(f"Grayscale images saved in: {output_dir}\n")
