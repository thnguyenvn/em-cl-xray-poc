import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def compute_per_class_metrics(
    y_true,
    y_prob,
    target_labels,
    threshold: float = 0.5,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []

    for i, label in enumerate(target_labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yscore = y_prob[:, i]

        row = {
            "label": label,
            "support": int(yt.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        }

        try:
            row["auc"] = roc_auc_score(yt, yscore)
        except Exception:
            row["auc"] = np.nan

        try:
            row["average_precision"] = average_precision_score(yt, yscore)
        except Exception:
            row["average_precision"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_global_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true.flatten(), y_pred.flatten()),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        metrics["auc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
    except Exception:
        metrics["auc_macro"] = np.nan

    try:
        metrics["average_precision_macro"] = average_precision_score(
            y_true,
            y_prob,
            average="macro",
        )
    except Exception:
        metrics["average_precision_macro"] = np.nan

    return metrics