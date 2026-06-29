"""
Dataset validation utilities for X-ray continual learning experiments.

This module is intentionally lightweight and reusable for:
- CheXpertDataset
- NIHChestXrayDataset
- future X-ray datasets
"""
import numpy as np
import os
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


class DatasetValidator:
    """
    Validate dataset structure, image paths, labels, and basic statistics.

    Expected dataset attributes:
    - dataset.df
    - dataset.target_labels
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

    def _resolve_path_from_row(self, row) -> str:
        """
        Resolve image path for supported dataset classes.
        """

        # CheXpert: has Path column
        if "Path" in row.index:
            img_path = row["Path"]

            if os.path.isabs(img_path):
                return img_path

            root_dir = getattr(self.dataset, "root_dir", "")
            return os.path.join(root_dir, img_path)

        # NIH: has Image Index column and dataset resolver
        if "Image Index" in row.index:
            image_name = row["Image Index"]

            if hasattr(self.dataset, "_resolve_image_path"):
                return self.dataset._resolve_image_path(image_name)

            root_dir = getattr(self.dataset, "root_dir", "")
            return os.path.join(root_dir, image_name)

        raise ValueError("Cannot resolve image path: unsupported dataframe format.")

    def validate_paths(
        self,
        max_check: Optional[int] = None,
        save_csv: bool = True,
    ) -> Dict:
        """
        Validate image file paths.

        Args:
            max_check:
                Maximum number of samples to check.
                If None, check the entire dataset.
            save_csv:
                Whether to save missing path report to CSV.

        Returns:
            Dictionary containing validation summary.
        """

        df = self.dataset.df
        total = len(df)

        if max_check is not None:
            df_check = df.iloc[:max_check]
        else:
            df_check = df

        missing_records: List[Dict] = []
        checked = len(df_check)

        for idx, row in tqdm(
            df_check.iterrows(),
            total=checked,
            desc=f"Checking paths: {self.dataset_name}",
        ):
            try:
                img_path = self._resolve_path_from_row(row)
                exists = os.path.exists(img_path)
            except Exception as exc:
                img_path = None
                exists = False
                error = str(exc)
            else:
                error = ""

            if not exists:
                missing_records.append(
                    {
                        "index": idx,
                        "image_path": img_path,
                        "error": error,
                    }
                )

        missing_count = len(missing_records)

        summary = {
            "dataset_name": self.dataset_name,
            "total_samples": total,
            "checked_samples": checked,
            "missing_files": missing_count,
            "missing_rate": missing_count / checked if checked > 0 else 0,
        }

        if save_csv:
            output_path = os.path.join(
                self.output_dir,
                f"{self.dataset_name}_missing_files.csv",
            )
            pd.DataFrame(missing_records).to_csv(output_path, index=False)
            summary["missing_file_report"] = output_path

        return summary
        
    def validate_labels(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Validate label statistics.

        This function works for both:
        - CheXpert labels: 1, 0, -1, NaN
        - NIH labels after preprocessing: 1, 0

        Args:
            save_csv:
                Whether to save label statistics to CSV.

        Returns:
            pandas.DataFrame
                Label-level statistics.
        """

        df = self.dataset.df
        rows = []

        for label in self.dataset.target_labels:
            if label not in df.columns:
                raise ValueError(f"Label '{label}' not found in dataset dataframe.")

            values = df[label]

            num_samples = len(values)
            positive = int((values == 1).sum())
            negative = int((values == 0).sum())
            uncertain = int((values == -1).sum())
            missing = int(values.isna().sum())

            positive_rate = positive / num_samples if num_samples > 0 else 0.0
            uncertain_rate = uncertain / num_samples if num_samples > 0 else 0.0
            missing_rate = missing / num_samples if num_samples > 0 else 0.0

            rows.append(
                {
                    "dataset_name": self.dataset_name,
                    "label": label,
                    "num_samples": num_samples,
                    "positive": positive,
                    "negative": negative,
                    "uncertain": uncertain,
                    "missing": missing,
                    "positive_rate": positive_rate,
                    "uncertain_rate": uncertain_rate,
                    "missing_rate": missing_rate,
                }
            )

        stats = pd.DataFrame(rows)

        if save_csv:
            output_file = os.path.join(
                self.output_dir,
                f"{self.dataset_name}_label_statistics.csv",
            )
            stats.to_csv(output_file, index=False)

        return stats
        
    def class_distribution(self, save_csv: bool = True) -> pd.DataFrame:
        """
        Compute class distribution from target labels.

        Returns:
            pandas.DataFrame
                One row per label with positive count and positive rate.
        """

        label_stats = self.validate_labels(save_csv=False)

        distribution = label_stats[
            [
                "dataset_name",
                "label",
                "positive",
                "positive_rate",
                "uncertain",
                "missing",
            ]
        ].copy()

        distribution = distribution.sort_values(
            by="positive",
            ascending=False,
        ).reset_index(drop=True)

        if save_csv:
            output_file = os.path.join(
                self.output_dir,
                f"{self.dataset_name}_class_distribution.csv",
            )
            distribution.to_csv(output_file, index=False)

        return distribution