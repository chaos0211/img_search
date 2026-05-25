from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.models.cbam import CBAMProxy


class SelfSimilarityTensorEncoder(nn.Module):
    def __init__(self, channels: int, neighborhood_size: int = 7):
        super().__init__()
        if neighborhood_size < 3 or neighborhood_size % 2 == 0:
            raise ValueError("neighborhood_size must be an odd integer greater than or equal to 3")
        self.channels = channels
        self.neighborhood_size = neighborhood_size
        block_count = (neighborhood_size - 1) // 2
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(1, 1, kernel_size=(1, 3, 3), bias=False),
                    nn.BatchNorm3d(1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(block_count)
            ]
        )
        self.restore = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, self_similarity_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, channels, neighborhood_area, height, width = self_similarity_tensor.shape
        if channels != self.channels:
            raise ValueError("self-similarity tensor channel count does not match encoder channels")
        if neighborhood_area != self.neighborhood_size * self.neighborhood_size:
            raise ValueError("self-similarity tensor neighborhood area does not match encoder window")

        encoded = self_similarity_tensor.reshape(
            batch_size,
            channels,
            self.neighborhood_size,
            self.neighborhood_size,
            height,
            width,
        )
        encoded = encoded.permute(0, 4, 5, 1, 2, 3).contiguous()
        encoded = encoded.reshape(batch_size * height * width, 1, channels, self.neighborhood_size, self.neighborhood_size)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = encoded.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return self.restore(encoded)


class SelfSimilarityEmbedding(nn.Module):
    architecture_name = "chen2025_self_similarity_embedding"

    def __init__(self, target_dim: int = 1024, channels: int = 2048, neighborhood_size: int = 7):
        super().__init__()
        self.target_dim = target_dim
        self.input_channels = channels
        self.channels = target_dim
        self.neighborhood_size = neighborhood_size
        self.radius = neighborhood_size // 2
        self.input_projection = (
            nn.Identity()
            if channels == target_dim
            else nn.Sequential(
                nn.Conv2d(channels, target_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_dim),
                nn.ReLU(inplace=True),
            )
        )
        self.attention = CBAMProxy(channels=target_dim)
        self.tensor_encoder = SelfSimilarityTensorEncoder(channels=target_dim, neighborhood_size=neighborhood_size)
        self.initial_norm = nn.BatchNorm2d(target_dim)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(target_dim, target_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_dim, target_dim, kernel_size=1, bias=False),
        )
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.gem_eps = 1e-6

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        initial_feature = self.input_projection(feature_map)
        attention_score = self.attention.attention_score(initial_feature)
        patches = F.unfold(initial_feature, kernel_size=self.neighborhood_size, padding=self.radius)
        batch_size, channels, height, width = initial_feature.shape
        patches = patches.reshape(batch_size, channels, self.neighborhood_size * self.neighborhood_size, height, width)
        self_similarity_tensor = attention_score.unsqueeze(2) * patches
        dense_self_similarity = self.tensor_encoder(self_similarity_tensor)
        embedded_feature = self.feature_fusion(dense_self_similarity + self.initial_norm(initial_feature))
        embedding = self._gem(embedded_feature)
        return F.normalize(embedding, dim=1)

    def config(self) -> dict[str, int | str]:
        return {
            "architecture": self.architecture_name,
            "input_channels": self.input_channels,
            "feature_channels": self.channels,
            "target_dim": self.target_dim,
            "neighborhood_size": self.neighborhood_size,
        }

    def _gem(self, feature_map: torch.Tensor) -> torch.Tensor:
        p = self.gem_p.clamp(min=1.0, max=6.0)
        pooled = F.avg_pool2d(feature_map.clamp(min=self.gem_eps).pow(p), kernel_size=feature_map.shape[-2:]).pow(1.0 / p)
        return pooled.flatten(1)
