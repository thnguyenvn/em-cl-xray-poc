import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
)


def plot_roc_curves(
    y_true,
    y_prob,
    target_labels,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))

    for i, label in enumerate(target_labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.3f}")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_pr_curves(
    y_true,
    y_prob,
    target_labels,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))

    for i, label in enumerate(target_labels):
        try:
            precision, recall, _ = precision_recall_curve(
                y_true[:, i],
                y_prob[:, i],
            )
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"{label} AP={pr_auc:.3f}")
        except Exception:
            continue

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_per_class_auc(
    per_class_csv,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(per_class_csv)

    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df["auc"])
    plt.ylabel("AUC")
    plt.title("Per-class AUC")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_training_history(
    history_csv,
    output_path,
    metric="train_loss",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(history_csv)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {history_csv}")

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df[metric], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Training history: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path