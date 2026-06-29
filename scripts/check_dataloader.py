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
from src.datasets import CheXpertDataset, get_train_transforms, get_eval_transforms
from src.utils import set_seed


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
        "--chexpert_root",
        type=str,
        default="/kaggle/input/chexpert-v10-small",
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default="/kaggle/input/chexpert-v10-small/train.csv",
    )

    parser.add_argument(
        "--valid_csv",
        type=str,
        default="/kaggle/input/chexpert-v10-small/valid.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/metrics",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    return parser.parse_args()


def summarize_dataset(name, dataset):
    df = dataset.df
    rows = []

    for label in dataset.target_labels:
        values = df[label]
        rows.append(
            {
                "dataset": name,
                "label": label,
                "num_samples": len(values),
                "positive": int((values == 1).sum()),
                "negative": int((values == 0).sum()),
                "uncertain": int((values == -1).sum()),
                "missing": int(values.isna().sum()),
                "positive_rate": float((values == 1).mean()),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.batch_size = args.batch_size

    set_seed(cfg.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DataLoader Validation")
    print("=" * 60)

    print("CheXpert root:", args.chexpert_root)
    print("Train CSV    :", args.train_csv)
    print("Valid CSV    :", args.valid_csv)

    train_dataset = CheXpertDataset(
        root_dir=args.chexpert_root,
        csv_file=args.train_csv,
        target_labels=TARGET_LABELS,
        transform=get_train_transforms(cfg.image_size),
        uncertainty_policy="zero",
    )

    valid_dataset = CheXpertDataset(
        root_dir=args.chexpert_root,
        csv_file=args.valid_csv,
        target_labels=TARGET_LABELS,
        transform=get_eval_transforms(cfg.image_size),
        uncertainty_policy="zero",
    )

    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    images, labels = next(iter(train_loader))

    print("Batch image shape:", tuple(images.shape))
    print("Batch label shape:", tuple(labels.shape))
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    assert images.ndim == 4, "Images should be NCHW."
    assert labels.ndim == 2, "Labels should be N x C."
    assert labels.shape[1] == len(TARGET_LABELS), "Label dimension mismatch."

    train_report = summarize_dataset("chexpert_train", train_dataset)
    valid_report = summarize_dataset("chexpert_valid", valid_dataset)

    report = pd.concat([train_report, valid_report], ignore_index=True)

    report_path = os.path.join(
        args.output_dir,
        "chexpert_dataloader_report.csv",
    )

    report.to_csv(report_path, index=False)

    print("Saved report:", report_path)

    print("=" * 60)
    print("DataLoader OK")
    print("=" * 60)


if __name__ == "__main__":
    main()