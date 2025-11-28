import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def run_pca(
        input_csv="Data/features/features_all.csv",
        figures_path="Results/figures/"
    ):
    """
    Corre PCA completo:
    - Normaliza los datos
    - PCA 2D y 3D
    - Varianza explicada
    Guarda gráficas en Results/figures/
    """

    print("\n📉 Starting PCA analysis...\n")

    os.makedirs(figures_path, exist_ok=True)

    # =============== LOAD FEATURES ===============
    df = pd.read_csv(input_csv)

    # Separar features numéricos y etiquetas
    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]
    y = df["species"]

    # =============== NORMALIZE ===============
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============== PCA FIT ===============
    pca = PCA(n_components=10)   # puedes subirlo si quieres más PCs
    pcs = pca.fit_transform(X_scaled)

    # Mostrar varianza explicada
    explained = pca.explained_variance_ratio_
    print("Variance ratio:", explained)

    # =============== PCA 2D ===============
    plt.figure(figsize=(8, 6))
    for species in np.unique(y):
        idx = y == species
        plt.scatter(pcs[idx, 0], pcs[idx, 1], label=species, s=20)

    plt.title("PCA - 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "pca_2d.png"))
    plt.close()

    print("✔ PCA 2D saved!")

    # =============== PCA 3D ===============
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for species in np.unique(y):
        idx = y == species
        ax.scatter(pcs[idx, 0], pcs[idx, 1], pcs[idx, 2], label=species, s=20)

    ax.set_title("PCA - 3D Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "pca_3d.png"))
    plt.close()

    print("✔ PCA 3D saved!")

    # =============== VARIANCE PLOT ===============
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(explained) * 100, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA - Cumulative Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "pca_variance.png"))
    plt.close()

    print("✔ PCA variance plot saved!\n")

    print("✨ PCA Completed Successfully!")
    return pcs, explained
