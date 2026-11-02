"""Evaluation metrics and figures (Task 4 + Task 8a)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)


def metrics_dict(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, pos_label=1), 4),
        "Recall": round(recall_score(y_true, y_pred, pos_label=1), 4),
        "F1": round(f1_score(y_true, y_pred, pos_label=1), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_proba), 4),
    }


def save_confusion_matrix(y_true, y_pred, name, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["N", "Y"], yticklabels=["N", "Y"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_roc_curves(results, path):
    """results: {name: (y_true, y_proba)}"""
    plt.figure(figsize=(5, 4.5))
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()