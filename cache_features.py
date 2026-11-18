"""Cache frozen backbone features for training and validation."""

import csv
import hashlib
import tempfile
from pathlib import Path

import torch

from image_batches import load_batch
from models import DEIT, RESNET, build_classifier, image_transform


@torch.inference_mode()
def extract(model, rows, transform, device):
    features = []
    labels = []
    for start in range(0, len(rows), 16):
        images, targets = load_batch(rows[start:start + 16], transform, device)
        features.append(model.extract_features(images).cpu())
        labels.append(targets.cpu())
    vectors = torch.cat(features)
    if vectors.shape != (len(rows), model.head.in_features) or not torch.isfinite(vectors).all():
        raise ValueError("Unexpected feature shape or non-finite values.")
    return {
        "image_ids": [row["image_id"] for row in rows],
        "labels": torch.cat(labels),
        "features": vectors,
    }


def main():
    manifest = Path("splits/manifest.csv")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    output = Path("local-data/features")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    transform = image_transform()
    for name in (DEIT, RESNET):
        model = build_classifier(name, 10).to(device).eval()
        cache = {"model": name, "manifest_sha256": digest}
        for split in ("train", "validation"):
            selected = [row for row in rows if row["split"] == split]
            cache[split] = extract(model, selected, transform, device)
            print(name, split, tuple(cache[split]["features"].shape))
        with tempfile.TemporaryDirectory(dir=output) as staging:
            path = Path(staging) / f"{name}.pt"
            torch.save(cache, path)
            path.replace(output / path.name)


if __name__ == "__main__":
    main()
