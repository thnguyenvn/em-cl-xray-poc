import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CheXpertDataset(Dataset):
    """
    CheXpert-v1.0-small dataset loader for PoC experiments.

    Supports:
    - loading image paths from train.csv or valid.csv
    - selecting target disease labels
    - handling uncertain labels
    """

    def __init__(
        self,
        root_dir,
        csv_file,
        target_labels,
        transform=None,
        uncertainty_policy="zero",
    ):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.target_labels = target_labels
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy

        self.df = pd.read_csv(csv_file)
        self.df = self._prepare_dataframe(self.df)

    def _prepare_dataframe(self, df):
        df = df.copy()

        for label in self.target_labels:
            if label not in df.columns:
                raise ValueError(f"Label '{label}' not found in CSV columns.")

        if self.uncertainty_policy == "zero":
            df[self.target_labels] = df[self.target_labels].replace(-1, 0)
        elif self.uncertainty_policy == "one":
            df[self.target_labels] = df[self.target_labels].replace(-1, 1)
        elif self.uncertainty_policy == "ignore":
            for label in self.target_labels:
                df = df[df[label] != -1]
        elif self.uncertainty_policy == "keep":
            pass
        else:
            raise ValueError(
                "uncertainty_policy must be one of: zero, one, ignore, keep"
            )

        df[self.target_labels] = df[self.target_labels].fillna(0)

        return df.reset_index(drop=True)


    def _resolve_image_path(self, img_path: str) -> str:
        """
        Resolve image path for different CheXpert folder layouts.
        """

        img_path = str(img_path).replace("\\", "/")

        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path

        candidates = [
            os.path.join(self.root_dir, img_path),
            os.path.join(
                self.root_dir,
                img_path.replace("CheXpert-v1.0-small/", ""),
            ),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Image not found. Tried: {candidates}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self._resolve_image_path(row["Path"])

        image = Image.open(img_path).convert("RGB")

        labels = row[self.target_labels].values.astype("float32")

        if self.transform is not None:
            image = self.transform(image)

        return image, labels
