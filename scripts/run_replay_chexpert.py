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
from src.datasets import (
    CheXpertDataset,
    get_train_transforms,
    get_eval_transforms,
    create_continual_task_stream,
    summarize_task_stream,
)
from src.models import DenseNet121Classifier
from src.continual import ReplayMemory, ReplayTrainer
from src.evaluator import XRayEvaluator, compute_global_metrics
from src.utils import set_seed, get_logger
from src.checkpoint import save_checkpoint


TARGET_LABELS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
]


TASKS = [
    ["No Finding"],
    ["Atelectasis"],
    ["Cardiomegaly"],
    ["Edema"],
    ["Pleural Effusion"],
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chexpert_root", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)

    parser.add_argument("--epochs_per_task", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--replay_batch_size", type=int, default=16)
    parser.add_argument("--memory_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--output_dir", type=str, default="outputs")

    return parser.parse_args()


def evaluate_after_task(
    model,
    valid_loader,
    device,
    task_id,
):
    evaluator = XRayEvaluator(
        model=model,
        device=device,
        threshold=0.5,
    )

    outputs = evaluator.predict(valid_loader)

    metrics = compute_global_metrics(
        outputs["y_true"],
        outputs["y_prob"],
        threshold=0.5,
    )

    row = {
        "task_id": task_id,
    }
    row.update(metrics)

    return row


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.output_dir = args.output_dir
    cfg.metrics_dir = os.path.join(args.output_dir, "metrics")
    cfg.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    cfg.log_dir = os.path.join(args.output_dir, "logs")

    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.learning_rate = args.learning_rate
    cfg.replay_buffer_size = args.memory_size
    cfg.replay_batch_size = args.replay_batch_size

    os.makedirs(cfg.metrics_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_seed(cfg.seed)

    logger = get_logger(
        name="replay_chexpert",
        log_dir=cfg.log_dir,
    )

    logger.info("Starting CheXpert replay continual learning experiment")

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

    task_stream = create_continual_task_stream(
        dataset=train_dataset,
        task_label_groups=TASKS,
        require_single_positive=False,
    )

    task_summary = summarize_task_stream(task_stream)

    task_summary_path = os.path.join(
        cfg.metrics_dir,
        "replay_task_summary.csv",
    )

    task_summary.to_csv(task_summary_path, index=False)

    logger.info(f"Task summary saved to {task_summary_path}")
    logger.info(task_summary)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DenseNet121Classifier(
        num_classes=len(TARGET_LABELS),
        pretrained=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    memory = ReplayMemory(
        memory_size=cfg.replay_buffer_size,
    )

    trainer = ReplayTrainer(
        model=model,
        device=cfg.device,
        optimizer=optimizer,
        memory=memory,
        replay_batch_size=cfg.replay_batch_size,
        logger=logger,
    )

    train_history = []
    eval_history = []

    for task_id, (task_name, task_subset) in enumerate(
        task_stream.items(),
        start=1,
    ):
        logger.info("=" * 60)
        logger.info(f"Training {task_name}")
        logger.info("=" * 60)

        task_loader = DataLoader(
            task_subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        for epoch in range(1, args.epochs_per_task + 1):
            row = trainer.train_one_task(
                task_loader,
                task_id=task_id,
            )

            row["epoch"] = epoch
            row["task_name"] = task_name

            train_history.append(row)

        eval_row = evaluate_after_task(
            model=model,
            valid_loader=valid_loader,
            device=cfg.device,
            task_id=task_id,
        )

        eval_row["task_name"] = task_name
        eval_row["memory_size"] = len(memory)

        eval_history.append(eval_row)

        logger.info(f"Eval after task {task_id}: {eval_row}")

        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=task_id,
            metrics=eval_row,
            cfg=cfg,
            checkpoint_dir=cfg.checkpoint_dir,
            experiment_name="replay_chexpert",
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    train_history_path = os.path.join(
        cfg.metrics_dir,
        "replay_train_history.csv",
    )

    eval_history_path = os.path.join(
        cfg.metrics_dir,
        "replay_eval_history.csv",
    )

    pd.DataFrame(train_history).to_csv(train_history_path, index=False)
    pd.DataFrame(eval_history).to_csv(eval_history_path, index=False)

    logger.info(f"Saved train history: {train_history_path}")
    logger.info(f"Saved eval history: {eval_history_path}")

    print("=" * 60)
    print("Replay CheXpert experiment completed")
    print("=" * 60)
    print("Saved:", train_history_path)
    print("Saved:", eval_history_path)
    print("Saved:", task_summary_path)


if __name__ == "__main__":
    main()