import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch


def _config_to_dict(cfg: Any) -> Dict:
    if cfg is None:
        return {}

    if is_dataclass(cfg):
        return asdict(cfg)

    if isinstance(cfg, dict):
        return cfg

    return vars(cfg)


def save_checkpoint(
    model,
    optimizer=None,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    cfg: Any = None,
    checkpoint_dir: str = "outputs/checkpoints",
    filename: Optional[str] = None,
    experiment_name: str = "experiment",
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"{experiment_name}_epoch_{epoch:03d}.pth"

    path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "config": _config_to_dict(cfg),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)

    return path


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    map_location: str = "cpu",
):
    checkpoint = torch.load(
    path,
    map_location=map_location,
    weights_only=False,
)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint