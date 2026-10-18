import os
import warnings
from pathlib import Path

import torch
from PIL import ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

from food_ai.config import (
    EVAL_RESIZE_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
    module=r"PIL\.Image",
)


class NonEmptyImageFolder(ImageFolder):
    """ImageFolder variant that ignores placeholder classes with no images."""

    @staticmethod
    def find_classes(directory):
        root = Path(directory)
        classes = sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir()
            and any(
                item.is_file() and item.suffix.lower() in IMG_EXTENSIONS
                for item in entry.rglob("*")
            )
        )
        if not classes:
            raise FileNotFoundError(f"Không có lớp nào chứa ảnh trong {directory}")
        return classes, {name: index for index, name in enumerate(classes)}


class PrepareImage:
    """Áp dụng EXIF orientation và bảo đảm ảnh có ba kênh RGB."""

    def __call__(self, image):
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocessing_metadata():
    return {
        "color": "RGB",
        "exif_orientation": True,
        "resize": EVAL_RESIZE_SIZE,
        "crop": "center",
        "crop_size": IMAGE_SIZE,
        "normalize": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
    }


def build_transforms():
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    train_transform = transforms.Compose([
        PrepareImage(),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.85, 1.0),
            ratio=(0.85, 1.15),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.05,
            hue=0.0,
        ),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        PrepareImage(),
        transforms.Resize(EVAL_RESIZE_SIZE, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def create_loaders(
    train_root,
    valid_root,
    test_root,
    batch_size,
    workers,
):
    train_transform, eval_transform = build_transforms()
    train_dataset = NonEmptyImageFolder(train_root, transform=train_transform)
    valid_dataset = NonEmptyImageFolder(valid_root, transform=eval_transform)
    test_dataset = NonEmptyImageFolder(test_root, transform=eval_transform)

    if train_dataset.classes != valid_dataset.classes:
        raise ValueError(
            "Nhãn train/valid không khớp: "
            f"{train_dataset.classes} != {valid_dataset.classes}"
        )
    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Nhãn train/test không khớp: "
            f"{train_dataset.classes} != {test_dataset.classes}"
        )

    use_cuda = torch.cuda.is_available()
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": use_cuda,
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=use_cuda, **options
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **options)
    test_loader = DataLoader(test_dataset, shuffle=False, **options)
    return train_loader, valid_loader, test_loader, train_dataset.classes
