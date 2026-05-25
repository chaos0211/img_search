from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class ResNet101Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = None
        try:
            weights = ResNet101_Weights.IMAGENET1K_V2
            model = resnet101(weights=weights)
        except Exception:
            model = resnet101(weights=None)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.transforms = weights.transforms() if weights else Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def preprocess(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transforms(image)
        return tensor.unsqueeze(0)

    def forward_features(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(image_tensor)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        pooled = self.avgpool(feature_map).flatten(1)
        return feature_map, pooled
