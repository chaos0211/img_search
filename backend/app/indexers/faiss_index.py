from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from backend.app.indexers.base import BaseIndexer
from backend.app.utils.metric_utils import normalize_matrix, normalize_vector

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class FaissFlatIPIndex(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.index = None
        self.dim = 0

    def build(self, vectors: np.ndarray) -> None:
        if faiss is None:
            raise RuntimeError("FAISS 未安装")
        matrix = np.ascontiguousarray(normalize_matrix(vectors).astype(np.float32))
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("向量数据为空")
        self.vectors = matrix
        self.dim = int(matrix.shape[1])
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("FAISS 索引未构建")
        limit = min(max(int(top_k), 1), int(self.index.ntotal))
        query = np.ascontiguousarray(normalize_vector(query_vector).reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(query, limit)
        return indices[0].astype(np.int64), scores[0].astype(np.float32)

    def save(self, path: Path) -> None:
        if faiss is None:
            raise RuntimeError("FAISS 未安装")
        if self.index is None:
            raise ValueError("FAISS 索引未构建")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        if faiss is None:
            raise RuntimeError("FAISS 未安装")
        if not path.exists():
            raise FileNotFoundError(path)
        self.index = faiss.read_index(str(path))
        self.dim = int(self.index.d)

    def metadata(self) -> dict:
        vector_count = int(self.index.ntotal) if self.index is not None else 0
        return {
            "library": "faiss",
            "index_method": "FlatIP",
            "vector_count": vector_count,
            "dimension": int(self.dim),
            "index_class": "IndexFlatIP",
            "metric": "inner_product",
            "normalized": True,
        }
