from .evaluator import XRayEvaluator

from .metrics import (
    compute_global_metrics,
    compute_per_class_metrics,
)

from .visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_per_class_auc,
    plot_training_history,
)

from .report import generate_baseline_report

__all__ = [
    "XRayEvaluator",

    "compute_global_metrics",
    "compute_per_class_metrics",

    "plot_roc_curves",
    "plot_pr_curves",
    "plot_per_class_auc",
    "plot_training_history",
    "generate_baseline_report",
    "build_accuracy_matrix",
    "summarize_continual_metrics",
    "plot_accuracy_matrix",
    "compute_average_accuracy",
    "compute_average_forgetting",
    "compute_backward_transfer",
]