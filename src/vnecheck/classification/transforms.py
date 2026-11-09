from __future__ import annotations

import random


class ResizePadEnhance:
    def __init__(self, size: int, contrast: float, sharpen: float) -> None:
        self.size = size
        self.contrast = contrast
        self.sharpen = sharpen

    def __call__(self, image):
        import cv2
        import numpy as np
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        ratio = min(self.size / width, self.size / height)
        resized = image.resize(
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        array = np.asarray(canvas, dtype=np.float32)
        array = np.clip((array - 127.5) * self.contrast + 127.5, 0, 255).astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.0
        smooth = cv2.filter2D(array, -1, kernel)
        enhanced = cv2.addWeighted(smooth, 1 - self.sharpen, array, self.sharpen, 0)
        return Image.fromarray(enhanced)


class TenPatchErasing:
    def __init__(self, regions: int, width_fraction: float, height_fraction: float, fill: float) -> None:
        self.regions = regions
        self.width_fraction = width_fraction
        self.height_fraction = height_fraction
        self.fill = fill

    def __call__(self, tensor):
        _, height, width = tensor.shape
        erase_width = max(1, round(width * self.width_fraction))
        erase_height = max(1, round(height * self.height_fraction))
        output = tensor.clone()
        for _ in range(self.regions):
            left = random.randint(0, width - erase_width)
            top = random.randint(0, height - erase_height)
            output[:, top : top + erase_height, left : left + erase_width] = self.fill
        return output


def build_transforms(config: dict, train: bool):
    from torchvision import transforms

    operations = [
        ResizePadEnhance(
            config["input_size"],
            config["contrast_enhancement"],
            config["sharpen_factor"],
        )
    ]
    if train:
        operations.extend(
            [
                transforms.RandomRotation(
                    config["rotation_degrees"],
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    fill=(114, 114, 114),
                ),
                transforms.ColorJitter(
                    brightness=tuple(config["brightness_range"]),
                    contrast=tuple(config["contrast_range"]),
                ),
                transforms.RandomHorizontalFlip(config["horizontal_flip_probability"]),
            ]
        )
    operations.append(transforms.ToTensor())
    erasing = config["random_erasing"]
    if train and erasing["enabled"]:
        operations.append(
            TenPatchErasing(
                erasing["regions"],
                erasing["width_fraction"],
                erasing["height_fraction"],
                erasing["fill"],
            )
        )
    operations.append(transforms.Normalize(config["normalization_mean"], config["normalization_std"]))
    return transforms.Compose(operations)
