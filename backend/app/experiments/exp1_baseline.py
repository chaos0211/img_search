from __future__ import annotations

from backend.app.services.offline_evaluation_service import OfflineEvaluationService


def run(gallery_manifest: str, query_manifest: str, top_k: int = 10):
    service = OfflineEvaluationService()
    return service.evaluate(gallery_manifest, query_manifest, index_type="brute", top_k=top_k, rerank=False, result_name="exp1_baseline.json")
