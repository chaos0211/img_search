from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, class_count: int, scale: float = 32.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(class_count, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings)
        normalized_weights = F.normalize(self.weight)
        cosine = F.linear(normalized_embeddings, normalized_weights)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        logits = cosine * (1.0 - one_hot) + target_logits * one_hot
        return logits * self.scale
