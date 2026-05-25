from __future__ import annotations

from backend.app.services.offline_evaluation_service import OfflineEvaluationService


def run(gallery_manifest: str, query_manifest: str, top_k: int = 10):
    service = OfflineEvaluationService()
    return {
        index_type: service.evaluate(gallery_manifest, query_manifest, index_type=index_type, top_k=top_k, rerank=False, result_name=f"exp4_{index_type}.json")
        for index_type in ("brute", "hnsw", "pq", "kd_tree")
    }
