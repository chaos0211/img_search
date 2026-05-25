from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseIndexer(ABC):
    def __init__(self):
        self.vectors: np.ndarray | None = None

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError

    def metadata(self) -> dict:
        vector_count = 0 if self.vectors is None else int(self.vectors.shape[0])
        dimension = 0 if self.vectors is None or self.vectors.ndim != 2 else int(self.vectors.shape[1])
        return {
            "vector_count": vector_count,
            "dimension": dimension,
        }

    def add(self, vectors: np.ndarray) -> None:
        self.build(vectors)

    def estimate_size(self, path: Path) -> int:
        return path.stat().st_size if path.exists() else 0
