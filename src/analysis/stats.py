import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_all_features(
        spatial_csv="Data/features/features_spatial.csv",
        fft_csv="Data/features/features_frequency.csv",
        lbp_csv="Data/features/features_lbp.csv",
        output_csv="Data/features/features_all.csv"
    ):
    """
    Fusiona los 3 archivos de características en un solo dataset multiclase.
    """

    print("\n📂 Loading all feature CSVs...")

    df_spatial = pd.read_csv(spatial_csv)
    df_fft = pd.read_csv(fft_csv)
    df_lbp = pd.read_csv(lbp_csv)

    # merge por filename + species
    df_merge = df_spatial.merge(df_fft, on=["filename", "species"], how="inner")
    df_merge = df_merge.merge(df_lbp, on=["filename", "species"], how="inner")

    # crear carpeta si no existe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_merge.to_csv(output_csv, index=False)

    print(f"✔ All features saved in: {output_csv}\n")

    return df_merge


def compute_statistics(df, tables_path="Results/tables/"):
    """
    Estadísticas descriptivas por especie (multiclase).
    """

    os.makedirs(tables_path, exist_ok=True)

    print("📊 Computing descriptive statistics...\n")

    # Solo columnas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns

    stats = df.groupby("species")[num_cols].agg(["mean", "std", "min", "max"])

    stats.to_csv(os.path.join(tables_path, "descriptive_statistics.csv"))

    print("✔ Descriptive statistics saved!\n")
    return stats


def compute_correlations(df, figures_path="Results/figures/"):
    """
    Calcula la matriz de correlaciones entre features y genera un heatmap.
    """

    os.makedirs(figures_path, exist_ok=True)
    os.makedirs("Results/tables/", exist_ok=True)

    print("📈 Generating correlation matrix...\n")

    num_cols = df.select_dtypes(include=[np.number]).columns

    # matriz de correlación
    corr = df[num_cols].corr()

    # guardar CSV
    corr.to_csv("Results/tables/correlation_matrix.csv")

    # heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap (Multiclass)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "correlation_heatmap.png"))
    plt.close()

    print("✔ Correlation heatmap saved!\n")
    return corr
