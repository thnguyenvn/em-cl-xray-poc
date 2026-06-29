import torch
import torch.nn as nn
from tqdm import tqdm


class MultilabelTrainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        criterion=None,
        logger=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.logger = logger

    def train_one_epoch(self, dataloader):
        self.model.train()

        total_loss = 0.0

        for images, labels in tqdm(dataloader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(dataloader.dataset)

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

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=1,
        metric_fn=None,
    ):
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
            }

            if val_loader is not None and metric_fn is not None:
                y_true, y_prob = self.evaluate(val_loader)
                metrics = metric_fn(y_true, y_prob)
                row.update(metrics)

            history.append(row)

            if self.logger:
                self.logger.info(row)
            else:
                print(row)

        return history