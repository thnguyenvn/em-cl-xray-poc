"""
Image transforms for CheXpert and NIH Chest X-ray datasets.

This module provides reusable torchvision transforms for:
- training
- validation / testing
- feature extraction
"""

from typing import Optional

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_xray_transforms(
    image_size: int = 224,
    train: bool = True,
    use_imagenet_norm: bool = True,
    horizontal_flip: bool = True,
    rotation_degrees: int = 7,
):
    """
    Build standard transforms for chest X-ray classification.

    Args:
        image_size:
            Output image size. Default is 224 for DenseNet/ResNet.
        train:
            If True, use light augmentation. If False, deterministic transform.
        use_imagenet_norm:
            Use ImageNet normalization for pretrained CNN backbones.
        horizontal_flip:
            Whether to apply random horizontal flip during training.
        rotation_degrees:
            Maximum random rotation angle for training augmentation.

    Returns:
        torchvision.transforms.Compose
    """

    transform_list = []

    if train:
        transform_list.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=rotation_degrees),
            ]
        )

        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.append(transforms.ToTensor())

    if use_imagenet_norm:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    return transforms.Compose(transform_list)


def get_train_transforms(
    image_size: int = 224,
    use_imagenet_norm: bool = True,
):
    """
    Training transforms for chest X-ray images.
    """

    return get_xray_transforms(
        image_size=image_size,
        train=True,
        use_imagenet_norm=use_imagenet_norm,
    )


def get_eval_transforms(
    image_size: int = 224,
    use_imagenet_norm: bool = True,
):
    """
    Validation / test transforms for chest X-ray images.
    """

    return get_xray_transforms(
        image_size=image_size,
        train=False,
        use_imagenet_norm=use_imagenet_norm,
    )


def get_feature_transforms(
    image_size: int = 224,
    use_imagenet_norm: bool = True,
):
    """
    Deterministic transforms used before feature extraction for EM/GMM.
    """

    return get_eval_transforms(
        image_size=image_size,
        use_imagenet_norm=use_imagenet_norm,
    )