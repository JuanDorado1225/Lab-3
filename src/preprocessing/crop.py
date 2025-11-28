import os
import cv2

def crop_images(input_dir, output_dir="Data/processed/cropped", crop_ratio=0.10):
    """
    Recorta el % inferior de cada imagen del dataset.
    Mantiene la estructura de carpetas por especie.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n✂️ Starting CROPPING...")
    print(f"Input  directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Crop ratio: {crop_ratio * 100}%\n")

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

            h, w = img.shape[:2]

            # Calcular altura nueva quitando el % inferior
            crop_h = int(h * (1 - crop_ratio))

            cropped_img = img[:crop_h, :]  # recorte arriba → abajo

            out_path = os.path.join(out_species_dir, file)
            cv2.imwrite(out_path, cropped_img)

        print(f"✔ Finished {species}")

    print("\n✨ CROPPING COMPLETED!")
    print(f"Cropped images saved in: {output_dir}\n")
