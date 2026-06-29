import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.configs import ExperimentConfig
from src.checkpoint import save_checkpoint, load_checkpoint
from src.datasets import (
    CheXpertDataset,
    NIHChestXrayDataset,
    get_train_transforms,
    get_eval_transforms,
    create_continual_task_stream,
)
from src.models import DenseNet121Classifier
from src.metrics import compute_multilabel_metrics
from src.trainer import MultilabelTrainer
from src.utils import set_seed, get_logger
from src.experiments import run_baseline
from src.evaluator import XRayEvaluator
from src.continual import ReplayMemory, ReplayTrainer

from src.evaluator import (
    XRayEvaluator,
    compute_per_class_metrics,
    compute_global_metrics,
)

print("=" * 60)
print("Framework OK")
print("=" * 60)