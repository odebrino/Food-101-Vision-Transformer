"""Load manifest rows as RGB images and project labels."""

from pathlib import Path

import torch
from PIL import Image

IMAGE_ROOT = Path("local-data/food-101/images")


def load_batch(rows, transform, device):
    images = []
    for row in rows:
        with Image.open(IMAGE_ROOT / (row["image_id"] + ".jpg")) as image:
            images.append(transform(image.convert("RGB")))
    labels = torch.tensor([int(row["label"]) for row in rows], device=device)
    return torch.stack(images).to(device), labels
