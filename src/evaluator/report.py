import os
import pandas as pd


def generate_baseline_report(
    global_metrics_csv: str,
    per_class_metrics_csv: str,
    output_path: str,
    dataset_name: str = "CheXpert-v1.0-small",
    model_name: str = "DenseNet121",
    checkpoint_name: str = "baseline_finetuning_epoch_001.pth",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    global_df = pd.read_csv(global_metrics_csv)
    per_class_df = pd.read_csv(per_class_metrics_csv)

    metrics = global_df.iloc[0].to_dict()

    lines = []

    lines.append("# Baseline Evaluation Report\n")
    lines.append("## Experiment Summary\n")
    lines.append(f"- Dataset: **{dataset_name}**\n")
    lines.append(f"- Model: **{model_name}**\n")
    lines.append(f"- Checkpoint: `{checkpoint_name}`\n")

    lines.append("\n## Global Metrics\n")
    lines.append("| Metric | Value |\n")
    lines.append("|---|---:|\n")

    for key, value in metrics.items():
        try:
            lines.append(f"| {key} | {float(value):.4f} |\n")
        except Exception:
            lines.append(f"| {key} | {value} |\n")

    lines.append("\n## Per-class Metrics\n")
    lines.append(per_class_df.to_markdown(index=False))
    lines.append("\n\n## Figures\n")
    lines.append("- `outputs/figures/baseline_roc_curve.png`\n")
    lines.append("- `outputs/figures/baseline_pr_curve.png`\n")
    lines.append("- `outputs/figures/baseline_per_class_auc.png`\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    return output_path