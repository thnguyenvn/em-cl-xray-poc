import numpy as np


def final_average_accuracy(acc_matrix):
    acc_matrix = np.asarray(acc_matrix, dtype=float)
    return np.nanmean(acc_matrix[-1])


def average_forgetting(acc_matrix):
    acc_matrix = np.asarray(acc_matrix, dtype=float)
    final_row = acc_matrix[-1]
    forgetting = []
    for task_id in range(acc_matrix.shape[1] - 1):
        previous_best = np.nanmax(acc_matrix[: -1, task_id])
        forgetting.append(previous_best - final_row[task_id])
    return float(np.nanmean(forgetting)) if forgetting else 0.0


def backward_transfer(acc_matrix):
    acc_matrix = np.asarray(acc_matrix, dtype=float)
    final_row = acc_matrix[-1]
    diagonal = np.diag(acc_matrix)
    values = []
    for task_id in range(len(diagonal) - 1):
        values.append(final_row[task_id] - diagonal[task_id])
    return float(np.nanmean(values)) if values else 0.0
