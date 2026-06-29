import torch
import torch.nn as nn
from tqdm import tqdm

from .replay_memory import ReplayMemory


class ReplayTrainer:
    """
    Replay trainer for multi-label continual learning.

    For each incoming batch:
    - train on current batch
    - optionally mix with replay batch
    - update replay memory
    """

    def __init__(
        self,
        model,
        device,
        optimizer,
        memory: ReplayMemory,
        criterion=None,
        replay_batch_size: int = 32,
        logger=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.memory = memory
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.replay_batch_size = replay_batch_size
        self.logger = logger

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _combine_current_and_replay(self, images, labels):
        """
        Combine current batch with sampled replay batch.
        """

        if len(self.memory) == 0:
            return images, labels

        replay_images, replay_labels = self.memory.sample(
            self.replay_batch_size
        )

        replay_images = replay_images.to(self.device)
        replay_labels = replay_labels.to(self.device)

        combined_images = torch.cat([images, replay_images], dim=0)
        combined_labels = torch.cat([labels, replay_labels], dim=0)

        return combined_images, combined_labels

    def train_one_task(
        self,
        dataloader,
        task_id: int = 0,
    ):
        """
        Train model on one continual task.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(
            dataloader,
            desc=f"Replay Training Task {task_id}",
        ):
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            train_images, train_labels = self._combine_current_and_replay(
                images,
                labels,
            )

            self.optimizer.zero_grad()

            logits = self.model(train_images)
            loss = self.criterion(logits, train_labels)

            loss.backward()
            self.optimizer.step()

            self.memory.add_batch(images, labels)

            batch_size = train_images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)

        row = {
            "task_id": task_id,
            "train_loss": avg_loss,
            "memory_size": len(self.memory),
        }

        self._log(row)

        return row