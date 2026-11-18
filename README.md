# Food-101 Vision Transformer

This project compares DeiT-Tiny and ResNet-18 on ten Food-101 categories. Both models use the image IDs in `splits/manifest.csv`. Each ID is a path relative to `food-101/images/`, with the `.jpg` extension omitted. The class map in `splits/classes.json` records both the project label and the original Food-101 label.

## Prepare the data

Use the project environment with `source .venv/bin/activate`. The split was generated with Python 3.12, Torchvision 0.29.2 and Pillow 13.0.0. Place the extracted [official Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) in `local-data/food-101/`, keeping its `meta/` and `images/` directories. From the repository root, run:

```sh
python prepare_split.py
```

The script selects the first ten official classes alphabetically. Within each class, it sorts the official training IDs and shuffles them with a single Python random generator seeded with 42. The first 100 images go to training and the following 25 to validation. All 250 official test images per class go to the project test set. The script checks split membership, unique IDs, labels and RGB decoding, and saves a labeled training preview to `local-data/training-preview.jpg`. Local image data stays outside version control.

The local dataset copy contains the official class list and partition metadata plus the 3,750 selected JPEGs retrieved from the [Food-101 archive mirror](https://www.kaggle.com/datasets/dansbecker/food-101). Use the manifest to access images in this subset. The preparation script also works with the full archive.

## Models

Both models use a frozen ImageNet-1k feature extractor and a trainable ten-class linear layer. DeiT-Tiny uses its 192-dimensional classification token, and ResNet-18 uses its 512-dimensional pooled features. The input pipeline resizes the shorter edge to 256 pixels with bilinear interpolation, center-crops to 224 pixels and applies the ImageNet mean and standard deviation shared by the two checkpoints.

The GPU setup used Python 3.12, PyTorch 2.11.0 with CUDA 12.8, Torchvision 0.26.0, Pillow 13.0.0 and timm 1.0.31. To reproduce it in an activated virtual environment, run:

```sh
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install timm==1.0.31 Pillow==13.0.0
```

The builders load the [DeiT-Tiny weights](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k/tree/80e968688553f219e4a86f940ed945a23709c16f) and [ResNet-18 IMAGENET1K_V1 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). For offline use, place the linked DeiT revision's `model.safetensors` at `local-data/pretrained/deit_tiny.safetensors` and cache the Torchvision weights on the machine.

## Check the models before training

Run `python verify_models.py` from the repository root. Both models receive the same seeded batch of 16 training images. The check performs one AdamW step on each new head, compares all backbone parameters and buffers, and verifies scores and labels after saving and reloading on the training batch and 16 validation images. The pilot weights are temporary.

On an RTX 4060 Laptop GPU, the pilot measured 317 images/s for DeiT-Tiny and 319 images/s for ResNet-18. Each measurement includes reading and transforming eight batches of 16 local training images, inference and copying features back to host memory, after two warm-up batches. At these rates, extracting features for 1,000 training and 250 validation images would take about four seconds per model. Model loading, writing features to disk and fitting the classifier are separate costs.
