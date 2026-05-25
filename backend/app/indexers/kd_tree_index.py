from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree

from backend.app.indexers.base import BaseIndexer
from backend.app.utils.metric_utils import normalize_matrix, normalize_vector


class KDTreeIndex(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.tree: KDTree | None = None

    def build(self, vectors: np.ndarray) -> None:
        self.vectors = normalize_matrix(vectors)
        self.tree = KDTree(self.vectors, leaf_size=32, metric="euclidean")

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.tree is None or self.vectors is None:
            raise ValueError("index not built")
        distances, indices = self.tree.query(normalize_vector(query_vector).reshape(1, -1), k=top_k)
        indices = indices[0].astype(np.int32)
        distances = distances[0]
        scores = 1.0 - (distances ** 2) / 2.0
        return indices, scores.astype(np.float32)

    def save(self, path: Path) -> None:
        if self.vectors is None:
            raise ValueError("index not built")
        np.save(path, self.vectors)

    def load(self, path: Path) -> None:
        self.vectors = np.load(path)
        self.tree = KDTree(self.vectors, leaf_size=32, metric="euclidean")

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "library": "sklearn",
            "index_method": "KDTree",
            "index_class": "KDTree",
            "metric": "euclidean_on_l2_normalized",
            "leaf_size": 32,
            "normalized": True,
        }
