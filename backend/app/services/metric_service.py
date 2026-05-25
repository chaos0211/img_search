from __future__ import annotations

import json
from pathlib import Path

from backend.app.config import settings
from backend.app.database import fetch_all, fetch_one
from backend.app.services.gallery_service import GalleryService


class MetricService:
    def summary(self) -> dict:
        log_rows = fetch_all(
            """
            SELECT index_type, COUNT(*) AS total_runs, AVG(elapsed_ms) AS avg_elapsed
            FROM retrieval_logs
            GROUP BY index_type
            ORDER BY index_type ASC
            """
        )
        metric_rows = fetch_all("SELECT metrics_json FROM retrieval_logs ORDER BY id DESC LIMIT 100")

        map_values = []
        recall_values = []
        for row in metric_rows:
            payload = json.loads(row["metrics_json"] or "{}")
            if payload.get("mapAtK") is not None:
                map_values.append(float(payload["mapAtK"]))
            if payload.get("recallAtK") is not None:
                recall_values.append(float(payload["recallAtK"]))

        index_sizes = {}
        index_paths = {
            "faiss": settings.index_root / "faiss" / "gallery.index",
            "brute": settings.index_root / "brute_index.npy",
            "kd_tree": settings.index_root / "kd_tree_index.npy",
            "hnsw": settings.index_root / "hnsw_index.bin",
            "pq": settings.index_root / "pq_index.npz",
        }
        for method, path in index_paths.items():
            index_sizes[method] = path.stat().st_size if path.exists() else 0

        return {
            "overview": {
                "averageMapAtK": round(sum(map_values) / len(map_values), 4) if map_values else None,
                "averageRecallAtK": round(sum(recall_values) / len(recall_values), 4) if recall_values else None,
                "queryCount": sum(int(row["total_runs"]) for row in log_rows),
                "galleryCount": int((fetch_one(f"SELECT COUNT(*) AS total FROM images WHERE {GalleryService._active_gallery_filter()}") or {}).get("total") or 0),
            },
            "methods": [
                {
                    "method": row["index_type"],
                    "runs": int(row["total_runs"]),
                    "averageElapsedMs": round(float(row["avg_elapsed"]), 3),
                    "indexSizeBytes": int(index_sizes.get(row["index_type"], 0)),
                }
                for row in log_rows
            ],
        }
