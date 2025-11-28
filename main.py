import sys
import os

# Añadir ruta raíz del proyecto al PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== IMPORTS DEL PIPELINE =====

# --- Preprocessing ---
from src.preprocessing.cleaning import clean_dataset
from src.preprocessing.crop import crop_images
from src.preprocessing.resize import resize_images
from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.enhancement import enhance_images

# --- Feature extraction ---
from src.analysis.features_spatial import extract_spatial_features
from src.analysis.features_fft import extract_frequency_features
from src.analysis.features_lbp import extract_lbp_features

# --- Stats / correlación ---
from src.analysis.stats import (
    load_all_features,
    compute_statistics,
    compute_correlations
)

# --- PCA ---
from src.analysis.pca import run_pca

# --- CLASSIFICATION + VISUALIZATIONS ---
from src.models.classifier import train_and_evaluate_models


def main():

    # ==================================
    #          SWITCHES DEL PIPELINE
    # ==================================

    # --- PREPROCESSING (solo si necesitas rehacer todo) ---
    RUN_CLEANING = True
    RUN_CROP = True
    RUN_RESIZE = True
    RUN_GRAYSCALE = True
    RUN_ENHANCEMENT = True

    # --- FEATURE EXTRACTION (se dejan en False si ya están hechos) ---
    RUN_FEATURE_SPATIAL = True
    RUN_FEATURE_FFT = True
    RUN_FEATURE_LBP = True

    # --- ANALYSIS (estadísticas + correlación) ---
    RUN_STATS = True
    RUN_CORR = True

    # --- PCA ---
    RUN_PCA = True

    # --- CLASSIFICATION + MODEL PLOTS ---
    RUN_CLASSIFICATION = True   

    # ==================================
    #             RUTAS
    # ==================================

    raw_data_path = "Data/raw/train_features"
    clean_path = "Data/processed/clean"
    cropped_path = "Data/processed/cropped"
    resized_path = "Data/processed/resized"
    grayscale_path = "Data/processed/grayscale"
    enhanced_path = "Data/processed/enhanced"

    # ==================================
    #               PIPELINE
    # ==================================

    # --- 1) CLEANING ---
    if RUN_CLEANING:
        clean_dataset(raw_data_path, clean_path)

    # --- 2) CROPPING ---
    if RUN_CROP:
        crop_images(clean_path, cropped_path)

    # --- 3) RESIZE ---
    if RUN_RESIZE:
        resize_images(cropped_path, resized_path, size=(256, 256))

    # --- 4) GRAYSCALE ---
    if RUN_GRAYSCALE:
        convert_to_grayscale(resized_path, grayscale_path)

    # --- 5) ENHANCEMENT ---
    if RUN_ENHANCEMENT:
        enhance_images(grayscale_path, enhanced_path)

    # --- 6) SPATIAL FEATURES ---
    if RUN_FEATURE_SPATIAL:
        extract_spatial_features(enhanced_path)

    # --- 7) FFT FEATURES ---
    if RUN_FEATURE_FFT:
        extract_frequency_features(enhanced_path)

    # --- 8) LBP FEATURES ---
    if RUN_FEATURE_LBP:
        extract_lbp_features(enhanced_path)

    # ==================================
    #   UNIR FEATURES PARA ANALYSIS / PCA / CLASSIFICATION
    # ==================================

    if RUN_STATS or RUN_CORR or RUN_PCA or RUN_CLASSIFICATION:
        df_all = load_all_features()

    # --- 9) DESCRIPTIVE STATISTICS ---
    if RUN_STATS:
        compute_statistics(df_all)

    # --- 10) CORRELATION MATRIX ---
    if RUN_CORR:
        compute_correlations(df_all)

    # --- 11) PCA ---
    if RUN_PCA:
        run_pca()

    # --- 12) CLASSIFICATION + PLOTS ---
    if RUN_CLASSIFICATION:
        train_and_evaluate_models()


# ==================================
#         ENTRY POINT
# ==================================
if __name__ == "__main__":
    main()
