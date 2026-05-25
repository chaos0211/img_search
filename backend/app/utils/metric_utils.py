from __future__ import annotations

from typing import Iterable

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def cosine_scores(query_vector: np.ndarray, gallery_vectors: np.ndarray) -> np.ndarray:
    normalized_query = normalize_vector(query_vector)
    normalized_gallery = normalize_matrix(gallery_vectors)
    return normalized_gallery @ normalized_query


def recall_at_k(query_label: str | None, retrieved_labels: Iterable[str | None], total_relevant: int) -> float | None:
    if not query_label or total_relevant <= 0:
        return None
    relevant_found = sum(1 for label in retrieved_labels if label == query_label)
    return round(relevant_found / total_relevant, 4)


def precision_at_k(query_label: str | None, retrieved_labels: Iterable[str | None], k: int | None = None) -> float | None:
    if not query_label:
        return None
    labels = list(retrieved_labels)
    if k is not None:
        labels = labels[: max(int(k), 0)]
    if not labels:
        return 0.0
    relevant_found = sum(1 for label in labels if label == query_label)
    return round(relevant_found / len(labels), 4)


def map_at_k(query_label: str | None, retrieved_labels: list[str | None], total_relevant: int) -> float | None:
    if not query_label or total_relevant <= 0:
        return None
    hit_count = 0
    precision_sum = 0.0
    for rank, label in enumerate(retrieved_labels, start=1):
        if label == query_label:
            hit_count += 1
            precision_sum += hit_count / rank
    if hit_count == 0:
        return 0.0
    return round(precision_sum / min(total_relevant, len(retrieved_labels)), 4)
