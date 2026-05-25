from __future__ import annotations

from backend.app.services.offline_evaluation_service import OfflineEvaluationService


def run(gallery_manifest: str):
    service = OfflineEvaluationService()
    thresholds = [round(value, 2) for value in [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]]
    return service.scan_duplicate_thresholds(gallery_manifest, thresholds)
