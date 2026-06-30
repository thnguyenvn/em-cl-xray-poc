import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.evaluator import (
    build_accuracy_matrix,
    summarize_continual_metrics,
    plot_accuracy_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_history_csv",
        type=str,
        default="outputs/metrics/replay_eval_history.csv",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="auc_macro",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    metrics_dir = os.path.join(args.output_dir, "metrics")
    figures_dir = os.path.join(args.output_dir, "figures")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    matrix_path = os.path.join(
        metrics_dir,
        f"replay_accuracy_matrix_{args.metric}.csv",
    )

    summary_path = os.path.join(
        metrics_dir,
        f"replay_continual_summary_{args.metric}.csv",
    )

    heatmap_path = os.path.join(
        figures_dir,
        f"replay_accuracy_matrix_{args.metric}.png",
    )

    matrix_df = build_accuracy_matrix(
        eval_history_csv=args.eval_history_csv,
        metric=args.metric,
        output_path=matrix_path,
    )

    summary_df = summarize_continual_metrics(
        accuracy_matrix_csv=matrix_path,
        output_path=summary_path,
    )

    plot_accuracy_matrix(
        accuracy_matrix_csv=matrix_path,
        output_path=heatmap_path,
        title=f"Replay Accuracy Matrix ({args.metric})",
    )

    print("=" * 60)
    print("Continual Replay Evaluation Completed")
    print("=" * 60)
    print(matrix_df)
    print(summary_df)
    print("Saved:", matrix_path)
    print("Saved:", summary_path)
    print("Saved:", heatmap_path)


if __name__ == "__main__":
    main()