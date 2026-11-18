"""Prepare and validate the shared ten-class Food-101 split."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from torchvision.datasets import Food101


def index_images(dataset, image_root, selected):
    groups = defaultdict(dict)
    for index, path in enumerate(dataset._image_files):
        name = path.parent.name
        if name in selected:
            image_id = path.relative_to(image_root).with_suffix("").as_posix()
            if image_id in groups[name]:
                raise ValueError(f"Duplicate official image ID: {image_id}")
            groups[name][image_id] = index
    return groups


def prepare(root, output):
    train = Food101(root, split="train")
    test = Food101(root, split="test")
    metadata = root / "food-101" / "meta"
    names = sorted((metadata / "classes.txt").read_text().splitlines())
    if len(names) != 101 or names != train.classes or names != test.classes:
        raise ValueError("Class names differ between the official metadata and loaders.")

    selected = names[:10]
    image_root = root / "food-101" / "images"
    train_ids = index_images(train, image_root, selected)
    test_ids = index_images(test, image_root, selected)
    rng = random.Random(42)
    counts = Counter()
    seen = set()
    rows = []
    class_map = []
    preview = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(preview)

    for label, name in enumerate(selected):
        original_id = train.class_to_idx[name]
        class_map.append(dict(label=label, class_name=name, food101_label=original_id))
        training = sorted(train_ids[name])
        testing = sorted(test_ids[name])
        if len(training) != 750 or len(testing) != 250 or set(training).intersection(testing):
            raise ValueError(f"Invalid official partitions for {name}.")
        rng.shuffle(training)
        for split, ids in (("train", training[:100]), ("validation", training[100:125]), ("test", testing)):
            dataset, index = (test, test_ids[name]) if split == "test" else (train, train_ids[name])
            for image_id in ids:
                if image_id in seen:
                    raise ValueError(f"Overlapping project splits: {image_id}")
                image, target = dataset[index[image_id]]
                if target != original_id or image.mode != "RGB" or min(image.size) == 0:
                    raise ValueError(f"Invalid label or RGB image: {image_id}")
                if split == "train" and image_id == ids[0]:
                    x, y = (label % 5) * 160, (label // 5) * 150
                    tile = ImageOps.contain(image, (156, 120))
                    preview.paste(tile, (x + (160 - tile.width) // 2, y))
                    draw.text((x + 4, y + 122), name.replace("_", " "), fill="black")
                seen.add(image_id)
                counts[split, label] += 1
                rows.append((image_id, name, label, original_id, split))

    for split, expected in (("train", 100), ("validation", 25), ("test", 250)):
        if any(counts[split, label] != expected for label in range(10)):
            raise ValueError(f"Unbalanced {split} partition.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("image_id", "class_name", "label", "food101_label", "split"))
        writer.writerows(rows)
    (output / "classes.json").write_text(json.dumps(class_map, indent=2) + "\n", encoding="utf-8")
    preview.save(root / "training-preview.jpg", quality=90)
    print(f"Verified {len(seen):,} unique RGB images across {len(selected)} classes.")
    for split in ("train", "validation", "test"):
        print(f"{split}: {sum(counts[split, label] for label in range(10)):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("local-data"), help="Parent of the food-101 directory")
    parser.add_argument("--output", type=Path, default=Path("splits"), help="Directory for the manifest and class map")
    args = parser.parse_args()
    prepare(args.root, args.output)
