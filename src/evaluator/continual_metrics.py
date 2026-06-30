import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_accuracy_matrix(
    eval_history_csv: str,
    metric: str = "auc_macro",
    output_path: str = "outputs/metrics/accuracy_matrix.csv",
):
    """
    Build a simple continual learning accuracy matrix from eval history.

    Current replay_eval_history.csv has one row per task.
    This creates a diagonal-style matrix as a first CL report.

    Later, when we evaluate all previous tasks after each training task,
    this function will support full T x T matrices.
    """

    df = pd.read_csv(eval_history_csv)

    task_ids = sorted(df["task_id"].unique())
    n_tasks = len(task_ids)

    matrix = np.full((n_tasks, n_tasks), np.nan)

    for _, row in df.iterrows():
        task_id = int(row["task_id"])
        matrix[task_id - 1, task_id - 1] = float(row[metric])

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"After_T{i}" for i in task_ids],
        columns=[f"Task_{i}" for i in task_ids],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matrix_df.to_csv(output_path)

    return matrix_df


def compute_average_accuracy(accuracy_matrix):
    """
    Average accuracy after the final task over all seen tasks.
    """

    matrix = np.asarray(accuracy_matrix, dtype=float)

    final_row = matrix[-1, :]

    return float(np.nanmean(final_row))


def compute_average_forgetting(accuracy_matrix):
    """
    Compute average forgetting.

    Forgetting for task j:
        max performance before final step - final performance

    Only valid when full matrix is available.
    With diagonal-only matrix, this will return NaN.
    """

    matrix = np.asarray(accuracy_matrix, dtype=float)

    n_tasks = matrix.shape[0]
    forgetting_values = []

    for j in range(n_tasks - 1):
        past = matrix[: n_tasks - 1, j]
        final = matrix[n_tasks - 1, j]

        if np.isnan(final) or np.all(np.isnan(past)):
            continue

        forgetting_values.append(np.nanmax(past) - final)

    if len(forgetting_values) == 0:
        return np.nan

    return float(np.nanmean(forgetting_values))


def compute_backward_transfer(accuracy_matrix):
    """
    Compute BWT.

    BWT = average final performance on previous tasks
          minus performance when each task was first learned.
    """

    matrix = np.asarray(accuracy_matrix, dtype=float)

    n_tasks = matrix.shape[0]
    values = []

    for j in range(n_tasks - 1):
        initial = matrix[j, j]
        final = matrix[n_tasks - 1, j]

        if np.isnan(initial) or np.isnan(final):
            continue

        values.append(final - initial)

    if len(values) == 0:
        return np.nan

    return float(np.nanmean(values))


def summarize_continual_metrics(
    accuracy_matrix_csv: str,
    output_path: str = "outputs/metrics/continual_summary.csv",
):
    matrix_df = pd.read_csv(accuracy_matrix_csv, index_col=0)

    matrix = matrix_df.values

    summary = {
        "average_accuracy": compute_average_accuracy(matrix),
        "average_forgetting": compute_average_forgetting(matrix),
        "backward_transfer": compute_backward_transfer(matrix),
    }

    summary_df = pd.DataFrame([summary])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df


def plot_accuracy_matrix(
    accuracy_matrix_csv: str,
    output_path: str = "outputs/figures/accuracy_matrix_heatmap.png",
    title: str = "Continual Learning Accuracy Matrix",
):
    matrix_df = pd.read_csv(accuracy_matrix_csv, index_col=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.imshow(matrix_df.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Metric value")
    plt.xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=45)
    plt.yticks(range(len(matrix_df.index)), matrix_df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path