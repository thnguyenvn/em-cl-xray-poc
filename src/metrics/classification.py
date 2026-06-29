import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_multilabel_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    """
    Compute common multi-label metrics.

    Parameters
    ----------
    y_true : ndarray

    y_prob : ndarray

    threshold : float

    Returns
    -------
    dict
    """

    y_true = np.asarray(y_true)

    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true.flatten(),
        y_pred.flatten(),
    )

    metrics["precision_micro"] = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["recall_micro"] = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["f1_micro"] = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["precision_macro"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["recall_macro"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["f1_macro"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    try:

        metrics["auc_macro"] = roc_auc_score(
            y_true,
            y_prob,
            average="macro",
        )

    except Exception:

        metrics["auc_macro"] = np.nan

    return metrics