import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(cm, classes, model_name, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multiclass_roc(y_test, y_score, classes, model_name, output_path):
    """
    ROC One-vs-Rest para cada clase
    """

    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall(y_test, y_score, classes, model_name, output_path):
    """
    Precision–Recall Curve para cada clase (One-vs-Rest)
    """

    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f"{classes[i]} (AP = {ap:.2f})")

    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_metrics(report, model_name, output_path):
    """
    Barplot para Accuracy / F1 / Precision / Recall por modelo
    """

    metrics = ["precision", "recall", "f1-score"]
    scores = []

    for m in metrics:
        scores.append(report["weighted avg"][m])

    plt.figure(figsize=(7, 5))
    sns.barplot(x=metrics, y=scores)
    plt.title(f"{model_name} - Metrics (Weighted Avg)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
