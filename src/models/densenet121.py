import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 classifier for chest X-ray multi-label classification.

    Supports:
    - ImageNet pretrained backbone
    - feature extraction for EM / prototype / replay
    - multi-label output logits
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.feature_dim = in_features

        self.classifier = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.extract_features(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits