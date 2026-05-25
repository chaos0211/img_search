from __future__ import annotations

import numpy as np

from backend.app.utils.metric_utils import normalize_matrix, normalize_vector


class RerankService:
    def rerank(
        self,
        query_vector: np.ndarray,
        gallery_vectors: np.ndarray,
        initial_indices: np.ndarray,
        initial_scores: np.ndarray,
        top_k: int,
        alpha: float = 2.0,
        candidate_count: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized_gallery = normalize_matrix(gallery_vectors)
        normalized_query = normalize_vector(query_vector)
        if len(initial_indices) == 0:
            return initial_indices, initial_scores

        candidate_count = min(candidate_count, len(initial_indices))
        selected_indices = initial_indices[:candidate_count]
        selected_scores = np.maximum(initial_scores[:candidate_count], 0.0)
        weights = np.power(selected_scores + 1e-6, alpha)
        weights = weights / np.sum(weights)

        expanded_query = normalized_query * 0.6 + np.sum(normalized_gallery[selected_indices] * weights[:, None], axis=0) * 0.4
        expanded_query = normalize_vector(expanded_query.astype(np.float32))

        reranked_scores = normalized_gallery @ expanded_query
        order = np.argsort(reranked_scores)[::-1][:top_k]
        return order.astype(np.int32), reranked_scores[order].astype(np.float32)
