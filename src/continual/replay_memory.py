import random
from typing import List, Tuple

import torch


class ReplayMemory:
    """
    Simple replay memory for continual learning.

    Stores image tensors and multi-label targets.

    Strategy:
    - reservoir-like random replacement
    - fixed memory size
    """

    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.images: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.n_seen = 0

    def __len__(self):
        return len(self.images)

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.detach().cpu()
        labels = labels.detach().cpu()

        for img, lab in zip(images, labels):
            self.n_seen += 1

            if len(self.images) < self.memory_size:
                self.images.append(img)
                self.labels.append(lab)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.memory_size:
                    self.images[j] = img
                    self.labels[j] = lab

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.images) == 0:
            raise ValueError("Replay memory is empty.")

        batch_size = min(batch_size, len(self.images))
        indices = random.sample(range(len(self.images)), batch_size)

        images = torch.stack([self.images[i] for i in indices])
        labels = torch.stack([self.labels[i] for i in indices])

        return images, labels

    def state_dict(self):
        return {
            "memory_size": self.memory_size,
            "images": self.images,
            "labels": self.labels,
            "n_seen": self.n_seen,
        }

    def load_state_dict(self, state):
        self.memory_size = state["memory_size"]
        self.images = state["images"]
        self.labels = state["labels"]
        self.n_seen = state["n_seen"]