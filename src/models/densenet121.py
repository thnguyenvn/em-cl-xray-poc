"""
DenseNet121 backbone for Chest X-ray Continual Learning.

Features
--------
- ImageNet pretrained
- Multi-label classification
- Feature extraction for Replay / EM
- Freeze / unfreeze backbone
- Optional dropout
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    pretrained : bool
        Use ImageNet pretrained weights.

    freeze_backbone : bool
        Freeze feature extractor.

    dropout : float
        Dropout before classifier.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()

        weights = (
            DenseNet121_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = densenet121(weights=weights)

        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = 1024

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            self.feature_dim,
            num_classes,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze DenseNet feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze DenseNet feature extractor."""
        for param in self.features.parameters():
            param.requires_grad = True

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract 1024-dimensional embedding.

        Returns
        -------
        Tensor
            Shape (N,1024)
        """

        x = self.features(x)

        x = torch.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x

    def classify(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classification head.
        """

        features = self.dropout(features)

        logits = self.classifier(features)

        return logits

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Forward.

        Parameters
        ----------
        return_features
            If True return logits and embeddings.
        """

        features = self.extract_features(x)

        logits = self.classify(features)

        if return_features:
            return logits, features

        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sigmoid probabilities.
        """

        logits = self.forward(x)

        return torch.sigmoid(logits)