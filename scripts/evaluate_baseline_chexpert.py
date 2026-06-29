import os
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.configs import ExperimentConfig
from src.datasets import CheXpertDataset, get_eval_transforms
from src.models import DenseNet121Classifier
from src.checkpoint import load_checkpoint
from src.evaluator import (
    XRayEvaluator,
    compute_global_metrics,
    compute_per_class_metrics,
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

    parser.add_argument("--chexpert_root", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--output_dir", type=str, default="outputs")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.output_dir = args.output_dir
    cfg.metrics_dir = os.path.join(args.output_dir, "metrics")

    os.makedirs(cfg.metrics_dir, exist_ok=True)

    device = cfg.device

    valid_dataset = CheXpertDataset(
        root_dir=args.chexpert_root,
        csv_file=args.valid_csv,
        target_labels=TARGET_LABELS,
        transform=get_eval_transforms(cfg.image_size),
        uncertainty_policy="zero",
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DenseNet121Classifier(
        num_classes=len(TARGET_LABELS),
        pretrained=False,
    )

    model, _, checkpoint_info = load_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=None,
        map_location=device,
    )

    evaluator = XRayEvaluator(
        model=model,
        device=device,
        threshold=args.threshold,
    )

    outputs = evaluator.predict(valid_loader)

    global_metrics = compute_global_metrics(
        outputs["y_true"],
        outputs["y_prob"],
        threshold=args.threshold,
    )

    per_class_metrics = compute_per_class_metrics(
        outputs["y_true"],
        outputs["y_prob"],
        target_labels=TARGET_LABELS,
        threshold=args.threshold,
    )

    global_path = os.path.join(
        cfg.metrics_dir,
        "baseline_global_metrics.csv",
    )

    per_class_path = os.path.join(
        cfg.metrics_dir,
        "baseline_per_class_metrics.csv",
    )

    pred_path = os.path.join(
        cfg.metrics_dir,
        "baseline_predictions.npz",
    )

    pd.DataFrame([global_metrics]).to_csv(global_path, index=False)
    per_class_metrics.to_csv(per_class_path, index=False)

    import numpy as np

    np.savez_compressed(
        pred_path,
        y_true=outputs["y_true"],
        y_prob=outputs["y_prob"],
        y_pred=outputs["y_pred"],
    )

    print("=" * 60)
    print("Baseline Evaluation Completed")
    print("=" * 60)
    print("Checkpoint epoch:", checkpoint_info.get("epoch"))
    print("Global metrics:", global_metrics)
    print("Saved:", global_path)
    print("Saved:", per_class_path)
    print("Saved:", pred_path)


if __name__ == "__main__":
    main()