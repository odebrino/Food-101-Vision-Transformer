"""Frozen ImageNet feature extractors for the Food-101 linear classifiers."""

from pathlib import Path

import timm
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

DEIT = "deit_tiny_patch16_224.fb_in1k"
RESNET = "resnet18"
DEIT_WEIGHTS = Path("local-data/pretrained/deit_tiny.safetensors")


class DishClassifier(nn.Module):
    def __init__(self, backbone, feature_count, class_count):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Linear(feature_count, class_count)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        return self.head(self.extract_features(images))


def build_classifier(name, class_count, pretrained=True):
    if name == DEIT:
        config = {"file": str(DEIT_WEIGHTS)} if pretrained and DEIT_WEIGHTS.exists() else None
        backbone = timm.create_model(
            DEIT, pretrained=pretrained, pretrained_cfg_overlay=config, num_classes=0
        )
        feature_count = backbone.num_features
    elif name == RESNET:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        feature_count = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown model: {name}")
    return DishClassifier(backbone, feature_count, class_count)


def image_transform():
    return ResNet18_Weights.IMAGENET1K_V1.transforms()
