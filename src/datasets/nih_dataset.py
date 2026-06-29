"""
NIH ChestX-ray14 dataset loader.

This module supports:
- loading image paths from Data_Entry_2017.csv
- selecting target labels
- converting multi-label disease strings to binary vectors
"""

import os
from typing import List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NIHChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 dataset loader.

    Expected CSV columns:
    - Image Index
    - Finding Labels

    Args:
        root_dir:
            Root directory containing NIH image folders.
        csv_file:
            Path to Data_Entry_2017.csv.
        target_labels:
            List of disease labels to keep.
        transform:
            torchvision transform.
        image_subdirs:
            Optional list of subdirectories to search for images.
    """

    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        target_labels: List[str],
        transform=None,
        image_subdirs: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.target_labels = target_labels
        self.transform = transform

        self.image_subdirs = image_subdirs or [
            "images",
            "images_001/images",
            "images_002/images",
            "images_003/images",
            "images_004/images",
            "images_005/images",
            "images_006/images",
            "images_007/images",
            "images_008/images",
            "images_009/images",
            "images_010/images",
            "images_011/images",
            "images_012/images",
        ]

        self.df = pd.read_csv(csv_file)
        self.df = self._prepare_dataframe(self.df)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_cols = ["Image Index", "Finding Labels"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in NIH CSV.")

        for label in self.target_labels:
            df[label] = df["Finding Labels"].apply(
                lambda x: 1.0 if label in str(x).split("|") else 0.0
            )

        return df.reset_index(drop=True)

    def _resolve_image_path(self, image_name: str) -> str:
        direct_path = os.path.join(self.root_dir, image_name)
        if os.path.exists(direct_path):
            return direct_path

        for subdir in self.image_subdirs:
            candidate = os.path.join(self.root_dir, subdir, image_name)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Image '{image_name}' not found under root_dir={self.root_dir}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row["Image Index"]
        img_path = self._resolve_image_path(image_name)

        image = Image.open(img_path).convert("RGB")
        labels = row[self.target_labels].values.astype("float32")

        if self.transform is not None:
            image = self.transform(image)

        return image, labels