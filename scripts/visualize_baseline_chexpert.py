import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.evaluator import (
    plot_roc_curves,
    plot_pr_curves,
    plot_per_class_auc,
    plot_training_history,
)


TARGET_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction_npz",
        type=str,
        default="outputs/metrics/baseline_predictions.npz",
    )

    parser.add_argument(
        "--per_class_csv",
        type=str,
        default="outputs/metrics/baseline_per_class_metrics.csv",
    )

    parser.add_argument(
        "--history_csv",
        type=str,
        default="outputs/metrics/baseline_finetuning_history.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.prediction_npz)

    y_true = data["y_true"]
    y_prob = data["y_prob"]

    roc_path = plot_roc_curves(
        y_true=y_true,
        y_prob=y_prob,
        target_labels=TARGET_LABELS,
        output_path=os.path.join(args.output_dir, "baseline_roc_curve.png"),
    )

    pr_path = plot_pr_curves(
        y_true=y_true,
        y_prob=y_prob,
        target_labels=TARGET_LABELS,
        output_path=os.path.join(args.output_dir, "baseline_pr_curve.png"),
    )

    auc_path = plot_per_class_auc(
        per_class_csv=args.per_class_csv,
        output_path=os.path.join(args.output_dir, "baseline_per_class_auc.png"),
    )

    loss_path = None

    if os.path.exists(args.history_csv):
        loss_path = plot_training_history(
            history_csv=args.history_csv,
            output_path=os.path.join(args.output_dir, "baseline_loss_curve.png"),
            metric="train_loss",
        )

    print("=" * 60)
    print("Baseline visualization completed")
    print("=" * 60)
    print("Saved:", roc_path)
    print("Saved:", pr_path)
    print("Saved:", auc_path)
    if loss_path:
        print("Saved:", loss_path)


if __name__ == "__main__":
    main()