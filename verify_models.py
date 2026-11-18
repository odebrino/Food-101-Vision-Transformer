"""Check head updates, checkpoint recovery and feature extraction time."""

import csv
import json
import random
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from image_batches import load_batch
from models import DEIT, RESNET, build_classifier, image_transform

BATCH_SIZE = 16
TIMED_BATCHES = 8


def check_update(model, images, labels):
    model.train()
    frozen = {key: value.clone() for key, value in model.backbone.state_dict().items()}
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0.01)
    scores = model(images)
    if scores.shape != (BATCH_SIZE, 10) or not torch.isfinite(scores).all():
        raise ValueError("Expected 16 rows of 10 finite class scores.")
    loss = F.cross_entropy(scores, labels)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite cross-entropy.")
    loss.backward()
    optimizer.step()
    if torch.equal(before, model.head.weight):
        raise RuntimeError("The classifier weights did not change.")
    for key, value in model.backbone.state_dict().items():
        if not torch.equal(frozen[key], value):
            raise RuntimeError(f"Frozen parameter or buffer changed: {key}")
    return loss.item()


@torch.no_grad()
def check_reload(name, model, images, path):
    model.eval()
    expected = model(images)
    torch.save(model.state_dict(), path)
    restored = build_classifier(name, model.head.out_features, pretrained=False).to(images.device)
    restored.load_state_dict(torch.load(path, weights_only=True, map_location=images.device))
    actual = restored.eval()(images)
    torch.testing.assert_close(actual, expected)
    if not torch.equal(actual.argmax(1), expected.argmax(1)):
        raise RuntimeError("Predicted labels changed after reloading.")


def time_features(model, rows, transform, device):
    for start in range(0, (TIMED_BATCHES + 2) * BATCH_SIZE, BATCH_SIZE):
        if start == 2 * BATCH_SIZE:
            begin = perf_counter()
        images, _ = load_batch(rows[start:start + BATCH_SIZE], transform, device)
        model.extract_features(images).cpu()
    return TIMED_BATCHES * BATCH_SIZE / (perf_counter() - begin)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name() if device.type == "cuda" else "CPU")
    classes = json.loads(Path("splits/classes.json").read_text())
    if [entry["label"] for entry in classes] != list(range(10)):
        raise ValueError("Expected ten ordered Food-101 labels.")
    with Path("splits/manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rng = random.Random(42)
    training = [row for row in rows if row["split"] == "train"]
    rng.shuffle(training)
    validation = [row for row in rows if row["split"] == "validation"]
    rng.shuffle(validation)
    transform = image_transform()
    images, labels = load_batch(training[:BATCH_SIZE], transform, device)
    val_images, _ = load_batch(validation[:BATCH_SIZE], transform, device)
    if not ((labels >= 0) & (labels < len(classes))).all():
        raise ValueError("Invalid training labels.")
    with tempfile.TemporaryDirectory() as directory:
        for name in (DEIT, RESNET):
            model = build_classifier(name, len(classes)).to(device)
            loss = check_update(model, images, labels)
            check_reload(name, model, torch.cat((images, val_images)), Path(directory) / "pilot.pt")
            rate = time_features(model, training, transform, device)
            print(f"{name}: 16 x 10 scores, loss {loss:.4f}, head update and reload verified.")
            print(f"  {rate:.1f} images/s; estimated {(len(training) + len(validation)) / rate:.1f} s "
                  "for training and validation features.")


if __name__ == "__main__":
    main()
