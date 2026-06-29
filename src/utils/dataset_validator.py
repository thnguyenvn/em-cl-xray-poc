import os
from typing import Dict, Optional

import pandas as pd


class DatasetValidator:
    """
    Lightweight dataset validator.

    Expected dataset attributes:
    - df
    - target_labels
    """

    def __init__(
        self,
        dataset,
        dataset_name: str = "dataset",
        output_dir: str = "outputs/dataset_validation",
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        if not hasattr(dataset, "df"):
            raise ValueError("Dataset must have attribute 'df'.")

        if not hasattr(dataset, "target_labels"):
            raise ValueError("Dataset must have attribute 'target_labels'.")

    def validate_labels(self, save_csv: bool = True) -> pd.DataFrame:
        rows = []
        df = self.dataset.df

        for label in self.dataset.target_labels:
            values = df[label]

            rows.append(
                {
                    "dataset_name": self.dataset_name,
                    "label": label,
                    "num_samples": len(values),
                    "positive": int((values == 1).sum()),
                    "negative": int((values == 0).sum()),
                    "uncertain": int((values == -1).sum()),
                    "missing": int(values.isna().sum()),
                    "positive_rate": float((values == 1).mean()),
                }
            )

        stats = pd.DataFrame(rows)

        if save_csv:
            path = os.path.join(
                self.output_dir,
                f"{self.dataset_name}_label_statistics.csv",
            )
            stats.to_csv(path, index=False)

        return stats

    def class_distribution(self, save_csv: bool = True) -> pd.DataFrame:
        stats = self.validate_labels(save_csv=False)

        dist = stats[
            [
                "dataset_name",
                "label",
                "positive",
                "positive_rate",
                "uncertain",
                "missing",
            ]
        ].copy()

        dist = dist.sort_values("positive", ascending=False)

        if save_csv:
            path = os.path.join(
                self.output_dir,
                f"{self.dataset_name}_class_distribution.csv",
            )
            dist.to_csv(path, index=False)

        return dist

    def validate_paths(
        self,
        max_check: Optional[int] = 1000,
        save_csv: bool = True,
    ) -> Dict:
        total = len(self.dataset.df)
        checked = min(total, max_check) if max_check is not None else total

        summary = {
            "dataset_name": self.dataset_name,
            "total_samples": total,
            "checked_samples": checked,
            "missing_files": None,
            "note": "Path validation is simplified in Framework v1.0.",
        }

        return summary

    def run_all(
        self,
        max_path_check: Optional[int] = 1000,
        save_csv: bool = True,
    ) -> Dict:
        return {
            "path_summary": self.validate_paths(
                max_check=max_path_check,
                save_csv=save_csv,
            ),
            "label_statistics": self.validate_labels(save_csv=save_csv),
            "class_distribution": self.class_distribution(save_csv=save_csv),
        }