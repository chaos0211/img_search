from __future__ import annotations

from backend.app.services.offline_evaluation_service import OfflineEvaluationService


def run(
    baseline_gallery_manifest: str,
    baseline_query_manifest: str,
    embedding_gallery_manifest: str,
    embedding_query_manifest: str,
    top_k: int = 10,
):
    service = OfflineEvaluationService()
    return {
        "baseline": service.evaluate(
            baseline_gallery_manifest,
            baseline_query_manifest,
            index_type="brute",
            top_k=top_k,
            rerank=False,
            result_name="exp2_baseline.json",
        ),
        "embedding": service.evaluate(
            embedding_gallery_manifest,
            embedding_query_manifest,
            index_type="brute",
            top_k=top_k,
            rerank=False,
            result_name="exp2_embedding.json",
        ),
    }
