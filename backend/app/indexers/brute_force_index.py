from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.indexers.base import BaseIndexer
from backend.app.utils.metric_utils import normalize_matrix, normalize_vector


class BruteForceIndex(BaseIndexer):
    def build(self, vectors: np.ndarray) -> None:
        self.vectors = normalize_matrix(vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise ValueError("index not built")
        scores = self.vectors @ normalize_vector(query_vector)
        order = np.argsort(scores)[::-1][:top_k]
        return order.astype(np.int32), scores[order].astype(np.float32)

    def save(self, path: Path) -> None:
        if self.vectors is None:
            raise ValueError("index not built")
        np.save(path, self.vectors)

    def load(self, path: Path) -> None:
        self.vectors = np.load(path)

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "library": "numpy",
            "index_method": "BruteForce",
            "index_class": "BruteForce",
            "metric": "cosine",
            "normalized": True,
        }
