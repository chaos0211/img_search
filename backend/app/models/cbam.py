from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMProxy(nn.Module):
    def __init__(self, channels: int = 2048, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 32)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_projection = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.spatial_norm = nn.BatchNorm2d(1)

    def channel_refine(self, feature_map: torch.Tensor) -> torch.Tensor:
        avg_channel = feature_map.mean(dim=(2, 3), keepdim=True)
        max_channel = feature_map.amax(dim=(2, 3), keepdim=True)
        channel_attention = torch.sigmoid(self.channel_mlp(avg_channel) + self.channel_mlp(max_channel))
        return feature_map * channel_attention

    def attention_score(self, feature_map: torch.Tensor) -> torch.Tensor:
        channel_refined = self.channel_refine(feature_map)
        spatial_score = F.softplus(self.spatial_norm(self.spatial_projection(channel_refined)))
        spatial_score = spatial_score / spatial_score.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return channel_refined * spatial_score

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.attention_score(feature_map)
