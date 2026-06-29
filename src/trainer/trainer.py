import os
from typing import Callable, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.checkpoint import save_checkpoint


class MultilabelTrainer:
    """
    Trainer for multi-label chest X-ray classification.

    Features:
    - BCEWithLogitsLoss
    - train / evaluate / fit
    - metric tracking
    - best checkpoint saving
    - CSV history export
    - logger support
    """

    def __init__(
        self,
        model,
        device,
        optimizer,
        criterion=None,
        logger=None,
        cfg=None,
        experiment_name: str = "baseline",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.logger = logger
        self.cfg = cfg
        self.experiment_name = experiment_name

        self.best_score = -float("inf")
        self.history = []

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def train_one_epoch(self, dataloader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(dataloader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)

        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.float()

            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu()

            all_labels.append(labels)
            all_probs.append(probs)

        y_true = torch.cat(all_labels, dim=0).numpy()
        y_prob = torch.cat(all_probs, dim=0).numpy()

        return y_true, y_prob

    def _get_checkpoint_dir(self):
        if self.cfg is not None and hasattr(self.cfg, "checkpoint_dir"):
            return self.cfg.checkpoint_dir
        return "outputs/checkpoints"

    def _get_metrics_dir(self):
        if self.cfg is not None and hasattr(self.cfg, "metrics_dir"):
            return self.cfg.metrics_dir
        return "outputs/metrics"

    def _save_history(self):
        metrics_dir = self._get_metrics_dir()
        os.makedirs(metrics_dir, exist_ok=True)

        history_path = os.path.join(
            metrics_dir,
            f"{self.experiment_name}_history.csv",
        )

        pd.DataFrame(self.history).to_csv(history_path, index=False)

        return history_path

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        metric_fn: Optional[Callable] = None,
        save_best: bool = True,
        monitor: str = "auc_macro",
    ):
        """
        Train model.

        Args:
            train_loader:
                Training DataLoader.
            val_loader:
                Validation DataLoader.
            epochs:
                Number of epochs.
            metric_fn:
                Function that receives y_true, y_prob and returns metric dict.
            save_best:
                Whether to save best checkpoint.
            monitor:
                Metric name used to determine best checkpoint.

        Returns:
            list[dict]
                Training history.
        """

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)

            row: Dict = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            metrics = {}

            if val_loader is not None and metric_fn is not None:
                y_true, y_prob = self.evaluate(val_loader)
                metrics = metric_fn(y_true, y_prob)
                row.update(metrics)

            if save_best and metrics:
                score = metrics.get(monitor)

                if score is not None and score > self.best_score:
                    self.best_score = score

                    checkpoint_path = save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=row,
                        cfg=self.cfg,
                        checkpoint_dir=self._get_checkpoint_dir(),
                        experiment_name=self.experiment_name,
                    )

                    row["checkpoint"] = checkpoint_path
                    row["best_score"] = self.best_score

            self.history.append(row)

            history_path = self._save_history()
            row["history_path"] = history_path

            self._log(f"Epoch {epoch}: {row}")

        return self.history