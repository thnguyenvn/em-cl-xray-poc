import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.evaluator import generate_baseline_report


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--global_metrics_csv",
        type=str,
        default="outputs/metrics/baseline_global_metrics.csv",
    )

    parser.add_argument(
        "--per_class_metrics_csv",
        type=str,
        default="outputs/metrics/baseline_per_class_metrics.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/reports/baseline_report.md",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    path = generate_baseline_report(
        global_metrics_csv=args.global_metrics_csv,
        per_class_metrics_csv=args.per_class_metrics_csv,
        output_path=args.output_path,
    )

    print("Baseline report saved:", path)


if __name__ == "__main__":
    main()