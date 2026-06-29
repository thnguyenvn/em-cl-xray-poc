import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets import CheXpertDataset, get_train_transforms, get_eval_transforms
from src.models import DenseNet121Classifier
from src.trainer import MultilabelTrainer
from src.metrics import compute_multilabel_metrics
from src.utils import set_seed, get_logger


DEFAULT_TARGET_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
]


def run_baseline(
    cfg,
    chexpert_root: str,
    train_csv: str,
    valid_csv: str,
    target_labels=None,
):
    target_labels = target_labels or DEFAULT_TARGET_LABELS

    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    logger = get_logger(
        name="baseline",
        log_dir=cfg.log_dir,
    )

    logger.info("Starting baseline fine-tuning experiment")
    logger.info(cfg)

    train_dataset = CheXpertDataset(
        root_dir=chexpert_root,
        csv_file=train_csv,
        target_labels=target_labels,
        transform=get_train_transforms(cfg.image_size),
        uncertainty_policy="zero",
    )

    valid_dataset = CheXpertDataset(
        root_dir=chexpert_root,
        csv_file=valid_csv,
        target_labels=target_labels,
        transform=get_eval_transforms(cfg.image_size),
        uncertainty_policy="zero",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = DenseNet121Classifier(
        num_classes=len(target_labels),
        pretrained=cfg.pretrained,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    trainer = MultilabelTrainer(
        model=model,
        device=cfg.device,
        optimizer=optimizer,
        logger=logger,
        cfg=cfg,
        experiment_name="baseline_finetuning",
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=cfg.epochs,
        metric_fn=compute_multilabel_metrics,
        save_best=True,
        monitor="auc_macro",
    )

    history_path = os.path.join(
        cfg.metrics_dir,
        "baseline_finetuning_history.csv",
    )

    pd.DataFrame(history).to_csv(history_path, index=False)

    logger.info(f"Saved baseline history to {history_path}")

    return history