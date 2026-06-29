"""
Utilities for building continual learning task streams.

This module supports two common PoC settings:

1. Label-incremental / disease-incremental split:
   Each task contains a subset of disease labels.

2. Domain-incremental split:
   Each task can represent a dataset/domain such as CheXpert or NIH.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset


DEFAULT_XRAY_TASKS = [
    ["No Finding"],
    ["Atelectasis"],
    ["Cardiomegaly"],
    ["Edema"],
    ["Pleural Effusion"],
]


def get_default_xray_tasks() -> List[List[str]]:
    """
    Return the default 5-task disease stream for PoC.
    """

    return DEFAULT_XRAY_TASKS


def build_label_to_index(target_labels: Sequence[str]) -> Dict[str, int]:
    """
    Map label name to column index in the dataset label vector.
    """

    return {label: idx for idx, label in enumerate(target_labels)}


def find_positive_indices(
    labels_matrix: np.ndarray,
    label_indices: Sequence[int],
    require_single_positive: bool = False,
) -> List[int]:
    """
    Find samples that are positive for at least one selected label.

    Args:
        labels_matrix:
            N x C binary label matrix.
        label_indices:
            Label column indices used for the current task.
        require_single_positive:
            If True, keep only samples with exactly one positive label
            across all labels. This is useful for a simplified PoC.

    Returns:
        List of sample indices.
    """

    selected = labels_matrix[:, label_indices]
    mask = selected.sum(axis=1) > 0

    if require_single_positive:
        mask = mask & (labels_matrix.sum(axis=1) == 1)

    return np.where(mask)[0].tolist()


def build_task_indices_from_labels(
    labels_matrix: np.ndarray,
    target_labels: Sequence[str],
    task_label_groups: Sequence[Sequence[str]],
    require_single_positive: bool = False,
) -> Dict[str, List[int]]:
    """
    Build sample indices for each continual learning task.

    Args:
        labels_matrix:
            N x C binary label matrix.
        target_labels:
            List of all label names corresponding to columns in labels_matrix.
        task_label_groups:
            Example:
                [
                    ["No Finding"],
                    ["Atelectasis"],
                    ["Cardiomegaly"],
                    ["Edema"],
                    ["Pleural Effusion"],
                ]
        require_single_positive:
            Whether to keep only single-positive samples.

    Returns:
        Dictionary:
            {
                "Task_1_No_Finding": [indices...],
                "Task_2_Atelectasis": [indices...],
                ...
            }
    """

    label_to_idx = build_label_to_index(target_labels)
    task_indices = {}

    for task_id, task_labels in enumerate(task_label_groups, start=1):
        missing = [label for label in task_labels if label not in label_to_idx]
        if missing:
            raise ValueError(f"Task {task_id} has labels not in target_labels: {missing}")

        label_indices = [label_to_idx[label] for label in task_labels]
        indices = find_positive_indices(
            labels_matrix=labels_matrix,
            label_indices=label_indices,
            require_single_positive=require_single_positive,
        )

        task_name = f"Task_{task_id}_" + "_".join(
            label.replace(" ", "_") for label in task_labels
        )
        task_indices[task_name] = indices

    return task_indices


def build_task_subsets(
    dataset: Dataset,
    task_indices: Dict[str, List[int]],
) -> Dict[str, Subset]:
    """
    Convert task index dictionary into PyTorch Subset objects.
    """

    return {
        task_name: Subset(dataset, indices)
        for task_name, indices in task_indices.items()
    }


def extract_labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """
    Extract labels from a dataset that stores labels in dataset.df
    and dataset.target_labels.

    This works with CheXpertDataset and NIHChestXrayDataset implemented
    in this project.
    """

    if not hasattr(dataset, "df") or not hasattr(dataset, "target_labels"):
        raise ValueError(
            "Dataset must have attributes 'df' and 'target_labels'."
        )

    labels = dataset.df[dataset.target_labels].values.astype("float32")
    return labels


def create_continual_task_stream(
    dataset: Dataset,
    task_label_groups: Sequence[Sequence[str]] = DEFAULT_XRAY_TASKS,
    require_single_positive: bool = False,
) -> Dict[str, Subset]:
    """
    Create continual learning task stream from a dataset.

    Example:
        task_stream = create_continual_task_stream(
            dataset=train_dataset,
            task_label_groups=[
                ["No Finding"],
                ["Atelectasis"],
                ["Cardiomegaly"],
                ["Edema"],
                ["Pleural Effusion"],
            ],
            require_single_positive=True,
        )
    """

    labels_matrix = extract_labels_from_dataset(dataset)

    task_indices = build_task_indices_from_labels(
        labels_matrix=labels_matrix,
        target_labels=dataset.target_labels,
        task_label_groups=task_label_groups,
        require_single_positive=require_single_positive,
    )

    return build_task_subsets(dataset, task_indices)


def summarize_task_stream(task_stream: Dict[str, Subset]) -> pd.DataFrame:
    """
    Return a small dataframe summarizing task names and sample counts.
    """

    rows = []

    for task_name, subset in task_stream.items():
        rows.append(
            {
                "task": task_name,
                "num_samples": len(subset),
            }
        )

    return pd.DataFrame(rows)