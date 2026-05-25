from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.config import settings
from backend.app.indexers.base import BaseIndexer
from backend.app.utils.file_utils import read_json, write_json
from backend.app.utils.metric_utils import normalize_matrix, normalize_vector

try:
    import hnswlib
except ImportError:  # pragma: no cover
    hnswlib = None


class HNSWIndex(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.index = None

    def build(self, vectors: np.ndarray) -> None:
        if hnswlib is None:
            raise RuntimeError("hnswlib not installed")
        self.vectors = normalize_matrix(vectors)
        dim = self.vectors.shape[1]
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=self.vectors.shape[0],
            ef_construction=settings.hnsw_ef_construction,
            M=settings.hnsw_m,
        )
        self.index.add_items(self.vectors, np.arange(self.vectors.shape[0]))
        self.index.set_ef(settings.hnsw_ef_search)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("index not built")
        indices, distances = self.index.knn_query(normalize_vector(query_vector), k=top_k)
        indices = indices[0].astype(np.int32)
        scores = 1.0 - distances[0]
        return indices, scores.astype(np.float32)

    def save(self, path: Path) -> None:
        if self.index is None or self.vectors is None:
            raise ValueError("index not built")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(path))
        write_json(path.with_suffix(".json"), self.metadata())

    def load(self, path: Path) -> None:
        if hnswlib is None:
            raise RuntimeError("hnswlib not installed")
        metadata = read_json(path.with_suffix(".json"))
        dimension = int(metadata.get("dimension") or metadata.get("dim"))
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        self.index.load_index(str(path), max_elements=int(metadata["vector_count"]))
        self.index.set_ef(int(metadata.get("ef_search") or settings.hnsw_ef_search))

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "library": "hnswlib",
            "index_method": "HNSW",
            "index_class": "HNSW",
            "metric": "cosine",
            "normalized": True,
            "M": int(settings.hnsw_m),
            "ef_construction": int(settings.hnsw_ef_construction),
            "ef_search": int(settings.hnsw_ef_search),
        }
