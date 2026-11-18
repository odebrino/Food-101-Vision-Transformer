# Food-101 Vision Transformer

This project compares DeiT-Tiny and ResNet-18 on ten Food-101 categories. Both models use the image IDs in `splits/manifest.csv`. Each ID is a path relative to `food-101/images/`, with the `.jpg` extension omitted. The class map in `splits/classes.json` records both the project label and the original Food-101 label.

## Prepare the data

Use the project environment with `source .venv/bin/activate`. To create it on another machine, use Python 3.12 with Torchvision 0.29.2 and Pillow 13.0.0. Place the extracted [official Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) in `local-data/food-101/`, keeping its `meta/` and `images/` directories. From the repository root, run:

```sh
python prepare_split.py
```

The script selects the first ten official classes alphabetically. Within each class, it sorts the official training IDs and shuffles them with a single Python random generator seeded with 42. The first 100 images go to training and the following 25 to validation. All 250 official test images per class go to the project test set. The script checks split membership, unique IDs, labels and RGB decoding, and saves a labeled training preview to `local-data/training-preview.jpg`. Local image data stays outside version control.

The local dataset copy contains the official class list and partition metadata plus the 3,750 selected JPEGs retrieved from the [Food-101 archive mirror](https://www.kaggle.com/datasets/dansbecker/food-101). Use the manifest to access images in this subset. The preparation script also works with the full archive.
