import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


def train_and_evaluate_models(
        input_csv="Data/features/features_all.csv",
        results_path_tables="Results/tables/",
        results_path_figures="Results/figures/"
    ):
    """
    Entrena modelos multiclase:
    - Random Forest
    - SVM (RBF)
    - Logistic Regression
    Evalúa y guarda métricas y matriz de confusión.
    """

    print("\n🤖 Starting MULTICLASS CLASSIFICATION...\n")

    os.makedirs(results_path_tables, exist_ok=True)
    os.makedirs(results_path_figures, exist_ok=True)

    # =====================================
    # 1. Load dataset
    # =====================================
    df = pd.read_csv(input_csv)

    feature_cols = df.select_dtypes(include=[np.number]).columns
    X = df[feature_cols]
    y = df["species"]

    # =====================================
    # 2. Train/test split
    # =====================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # =====================================
    # 3. NORMALIZATION
    # =====================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================
    # 4. CLASSIFIERS
    # =====================================
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000, multi_class="multinomial")
    }

    results = {}

    for name, model in models.items():
        print(f"🏋️ Training model: {name}...")
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)

        # Reporte de clasificación
        report = classification_report(y_test, preds, output_dict=True)
        results[name] = report

        # Guardar reporte en CSV
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(results_path_tables, f"{name}_classification_report.csv"))

        # Matriz de confusión
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path_figures, f"{name}_confusion_matrix.png"))
        plt.close()

        print(f"✔ Finished: {name}")

    print("\n✨ Classification Completed!\n")
    return results
