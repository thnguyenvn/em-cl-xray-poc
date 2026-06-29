from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    num_classes: int = 5
    pretrained: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    metrics_dir: str = "outputs/metrics"
    figure_dir: str = "outputs/figures"

    replay_buffer_size: int = 1000
    replay_batch_size: int = 32

    em_components: int = 5
    em_max_iter: int = 100
    em_tol: float = 1e-3