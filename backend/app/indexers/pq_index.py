from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from backend.app.config import settings
from backend.app.indexers.base import BaseIndexer
from backend.app.utils.metric_utils import normalize_matrix, normalize_vector


class PQIndex(BaseIndexer):
    def __init__(self, subvectors: int | None = None, cluster_count: int | None = None):
        super().__init__()
        self.subvectors = subvectors or settings.pq_subvectors
        self.cluster_count = cluster_count or settings.pq_clusters
        self.codebooks: list[np.ndarray] = []
        self.codes: np.ndarray | None = None
        self.is_trained = False

    def build(self, vectors: np.ndarray) -> None:
        normalized = normalize_matrix(vectors)
        self.vectors = normalized
        self.train(normalized)
        self.add(normalized)

    def train(self, vectors: np.ndarray) -> None:
        normalized = normalize_matrix(vectors)
        vector_dim = normalized.shape[1]
        subvectors = min(self.subvectors, vector_dim)
        while vector_dim % subvectors != 0 and subvectors > 1:
            subvectors -= 1
        self.subvectors = subvectors
        sub_dim = vector_dim // self.subvectors

        codebooks: list[np.ndarray] = []
        for subspace in range(self.subvectors):
            start = subspace * sub_dim
            end = start + sub_dim
            chunk = normalized[:, start:end]
            cluster_count = min(self.cluster_count, max(2, chunk.shape[0]))
            kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init=3, batch_size=256)
            kmeans.fit(chunk)
            codebooks.append(kmeans.cluster_centers_.astype(np.float32))

        self.codebooks = codebooks
        self.codes = None
        self.is_trained = True

    def add(self, vectors: np.ndarray) -> None:
        if not self.is_trained or not self.codebooks:
            raise ValueError("PQ index not trained")
        normalized = normalize_matrix(vectors)
        self.vectors = normalized
        vector_dim = normalized.shape[1]
        sub_dim = vector_dim // self.subvectors
        max_clusters = max(codebook.shape[0] for codebook in self.codebooks)
        code_dtype = np.uint16 if max_clusters > np.iinfo(np.uint8).max + 1 else np.uint8
        codes = np.zeros((normalized.shape[0], self.subvectors), dtype=code_dtype)
        for subspace, codebook in enumerate(self.codebooks):
            start = subspace * sub_dim
            end = start + sub_dim
            chunk = normalized[:, start:end]
            distances = np.sum((chunk[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
            codes[:, subspace] = np.argmin(distances, axis=1).astype(code_dtype)
        self.codes = codes

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.codes is None or not self.codebooks:
            raise ValueError("index not built")
        query = normalize_vector(query_vector)
        vector_dim = query.shape[0]
        sub_dim = vector_dim // self.subvectors
        distance_tables = []
        for subspace, codebook in enumerate(self.codebooks):
            start = subspace * sub_dim
            end = start + sub_dim
            chunk = query[start:end]
            distances = np.sum((codebook - chunk) ** 2, axis=1)
            distance_tables.append(distances)

        approx_distances = np.zeros(self.codes.shape[0], dtype=np.float32)
        for subspace, distance_table in enumerate(distance_tables):
            approx_distances += distance_table[self.codes[:, subspace]]

        order = np.argsort(approx_distances)[:top_k]
        scores = 1.0 / (1.0 + approx_distances[order])
        return order.astype(np.int32), scores.astype(np.float32)

    def save(self, path: Path) -> None:
        if self.codes is None or not self.codebooks:
            raise ValueError("index not built")
        np.savez(
            path,
            subvectors=np.array([self.subvectors], dtype=np.int32),
            cluster_count=np.array([self.cluster_count], dtype=np.int32),
            codes=self.codes,
            **{f"codebook_{index}": codebook for index, codebook in enumerate(self.codebooks)},
        )

    def load(self, path: Path) -> None:
        payload = np.load(path)
        self.subvectors = int(payload["subvectors"][0])
        if "cluster_count" in payload:
            self.cluster_count = int(payload["cluster_count"][0])
        self.codes = payload["codes"]
        self.codebooks = [payload[f"codebook_{index}"] for index in range(self.subvectors)]
        self.is_trained = True

    def metadata(self) -> dict:
        codebook_size = int(sum(codebook.nbytes for codebook in self.codebooks))
        codes_size = 0 if self.codes is None else int(self.codes.nbytes)
        return {
            **super().metadata(),
            "library": "sklearn",
            "index_method": "PQ",
            "index_class": "PQ",
            "metric": "approx_l2_on_l2_normalized",
            "normalized": True,
            "subvectors": int(self.subvectors),
            "cluster_count": int(self.cluster_count),
            "trained": bool(self.is_trained),
            "added": self.codes is not None,
            "codebook_size_bytes": codebook_size,
            "codes_size_bytes": codes_size,
        }
