from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import numpy as np

from backend.app.config import settings
from backend.app.indexers.brute_force_index import BruteForceIndex
from backend.app.indexers.faiss_index import FaissFlatIPIndex
from backend.app.indexers.hnsw_index import HNSWIndex
from backend.app.indexers.kd_tree_index import KDTreeIndex
from backend.app.indexers.pq_index import PQIndex
from backend.app.services.gallery_service import GalleryService
from backend.app.services.rerank_service import RerankService
from backend.app.utils.attribute_schema import image_attributes


METADATA_VERSION = 2


class VectorIndexService:
    def __init__(self):
        self.gallery_service = GalleryService()
        self.rerank_service = RerankService()
        self.index_dir = settings.index_root / "faiss"
        self.index_path = self.index_dir / "gallery.index"
        self.metadata_path = self.index_dir / "gallery_metadata.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._indexer: FaissFlatIPIndex | None = None
        self._metadata: dict[str, Any] | None = None
        self._runtime_cache: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        metadata = self._load_metadata()
        return {
            "engine": "FAISS",
            "metric": "InnerProduct",
            "indexType": "IndexFlatIP",
            "vectorCount": int(metadata.get("vectorCount") or 0),
            "dimension": int(metadata.get("dimension") or 0),
            "skippedCount": int(metadata.get("skippedCount") or 0),
            "updatedAt": metadata.get("updatedAt"),
            "indexPath": str(self.index_path.relative_to(settings.project_root)) if self.index_path.exists() else None,
            "metadataPath": str(self.metadata_path.relative_to(settings.project_root)) if self.metadata_path.exists() else None,
            "indexSizeBytes": self.index_path.stat().st_size if self.index_path.exists() else 0,
            "ready": self.index_path.exists() and int(metadata.get("vectorCount") or 0) > 0,
            "metadataVersion": int(metadata.get("metadataVersion") or 0),
        }

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        excluded_image_id: int | None = None,
        method: str = "faiss",
        rerank: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        method = self._normalize_method(method)
        if method != "faiss" or rerank:
            return self._runtime_search(query_vector, top_k, excluded_image_id, method, rerank)

        self.ensure_ready()
        if self._indexer is None or self._metadata is None:
            raise ValueError("向量索引不可用")
        items = self._metadata.get("items") or []
        if not items:
            raise ValueError("图库为空")
        limit = max(int(top_k), 1)
        search_k = min(len(items), limit + (1 if excluded_image_id else 0) + 8)
        indices, scores = self._indexer.search(query_vector, search_k)
        matches: list[dict[str, Any]] = []
        for raw_index, score in zip(indices.tolist(), scores.tolist()):
            if raw_index < 0 or raw_index >= len(items):
                continue
            item = items[raw_index]
            if excluded_image_id and int(item["id"]) == int(excluded_image_id):
                continue
            matches.append({"item": item, "score": float(score), "rankIndex": int(raw_index)})
            if len(matches) >= limit:
                break
        return matches, items, self.status()

    def _runtime_search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        excluded_image_id: int | None,
        method: str,
        rerank: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        signature = self.gallery_service.feature_signature()
        cache_key = f"{method}:{signature}"
        cache = self._runtime_cache.get(cache_key)
        if cache is None:
            raw_items = self.gallery_service.load_feature_rows(include_thumbnails=False)
            items, matrix, skipped = self._select_vectors(raw_items)
            if matrix is None or not items:
                raise ValueError("图库为空")
            indexer = self._indexer_for(method)
            indexer.build(matrix)
            cache = {
                "items": items,
                "matrix": matrix,
                "indexer": indexer,
                "skipped": skipped,
                "signature": signature,
            }
            self._runtime_cache = {cache_key: cache}

        items = cache["items"]
        matrix = cache["matrix"]
        indexer = cache["indexer"]
        if int(np.asarray(query_vector).reshape(-1).shape[0]) != int(matrix.shape[1]):
            raise ValueError("查询特征与图库特征维度不一致")

        limit = max(int(top_k), 1)
        candidate_count = min(len(items), max(limit + (1 if excluded_image_id else 0) + 8, 20 if rerank else limit))
        indices, scores = indexer.search(query_vector, candidate_count)
        if rerank:
            indices, scores = self.rerank_service.rerank(query_vector, matrix, indices, scores, candidate_count)

        matches: list[dict[str, Any]] = []
        for raw_index, score in zip(indices.tolist(), scores.tolist()):
            if raw_index < 0 or raw_index >= len(items):
                continue
            item = items[raw_index]
            if excluded_image_id and int(item["id"]) == int(excluded_image_id):
                continue
            matches.append({"item": item, "score": float(score), "rankIndex": int(raw_index)})
            if len(matches) >= limit:
                break

        status = {
            **self.status(),
            "engine": self._method_label(method),
            "indexType": method,
            "metric": "Cosine",
            "vectorCount": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]),
            "skippedCount": int(cache["skipped"]),
            "indexSizeBytes": int(matrix.nbytes),
            "ready": True,
            "rerank": bool(rerank),
        }
        return matches, items, status

    def rebuild_gallery_index(self) -> dict[str, Any]:
        items = self.gallery_service.load_feature_rows(include_thumbnails=False)
        selected_items, matrix, skipped = self._select_vectors(items)
        signature = self.gallery_service.feature_signature()
        if matrix is None or not selected_items:
            self._indexer = None
            metadata = self._metadata_payload([], 0, skipped, signature)
            self._save_metadata(metadata)
            if self.index_path.exists():
                self.index_path.unlink()
            return self.status()

        indexer = FaissFlatIPIndex()
        indexer.build(matrix)
        indexer.save(self.index_path)

        metadata = self._metadata_payload(selected_items, int(matrix.shape[1]), skipped, signature)
        self._save_metadata(metadata)
        self._indexer = indexer
        self._metadata = metadata
        return self.status()

    def ensure_ready(self) -> dict[str, Any]:
        signature = self.gallery_service.feature_signature()
        metadata = self._load_metadata()
        if not self.index_path.exists() or metadata.get("signature") != signature or metadata.get("metadataVersion") != METADATA_VERSION:
            return self.rebuild_gallery_index()
        if self._indexer is None:
            indexer = FaissFlatIPIndex()
            indexer.load(self.index_path)
            self._indexer = indexer
        self._metadata = metadata
        return self.status()

    def _select_vectors(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray | None, int]:
        buckets: dict[int, list[tuple[dict[str, Any], np.ndarray]]] = {}
        skipped = 0
        for item in items:
            vector = np.asarray(item.get("feature"), dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                skipped += 1
                continue
            buckets.setdefault(int(vector.shape[0]), []).append((item, vector))
        if not buckets:
            return [], None, skipped

        target_dim = settings.feature_dim if settings.feature_dim in buckets else max(buckets, key=lambda dim: len(buckets[dim]))
        selected = buckets[target_dim]
        skipped += sum(len(values) for dim, values in buckets.items() if dim != target_dim)
        payload_items = [self._serialize_item(item) for item, _ in selected]
        matrix = np.stack([vector for _, vector in selected], axis=0).astype(np.float32)
        return payload_items, matrix, skipped

    @staticmethod
    def _serialize_item(item: dict[str, Any]) -> dict[str, Any]:
        attrs = image_attributes(item.get("labelName"))
        return {
            "id": int(item["id"]),
            "originalName": item.get("originalName"),
            "labelName": item.get("labelName"),
            "basicCategory": attrs.basic_category,
            "superCategory": attrs.super_category,
            "objectType": attrs.object_type,
            "clusterTag": attrs.cluster_tag,
            "attributeValues": attrs.attribute_values,
            "attributes": attrs.display_tags,
            "thumbnailPath": item.get("thumbnailPath"),
            "source": item.get("source"),
            "createdAt": item.get("createdAt"),
            "featurePath": item.get("featurePath"),
            "featureSize": item.get("featureSize"),
            "featureUpdatedAt": item.get("featureUpdatedAt"),
        }

    @staticmethod
    def _signature(items: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for item in items:
            digest.update(str(item.get("id")).encode("utf-8"))
            digest.update(b"|")
            digest.update(str(item.get("featurePath") or "").encode("utf-8"))
            digest.update(b"|")
            digest.update(str(item.get("featureSize") or "").encode("utf-8"))
            digest.update(b"|")
            digest.update(str(item.get("featureUpdatedAt") or "").encode("utf-8"))
            digest.update(b"|")
            digest.update(str(item.get("createdAt") or "").encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _metadata_payload(items: list[dict[str, Any]], dimension: int, skipped: int, signature: str) -> dict[str, Any]:
        return {
            "metadataVersion": METADATA_VERSION,
            "engine": "FAISS",
            "indexType": "IndexFlatIP",
            "metric": "InnerProduct",
            "dimension": int(dimension),
            "vectorCount": len(items),
            "skippedCount": int(skipped),
            "items": items,
            "signature": signature,
            "updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.exists():
            return {}
        try:
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            self._metadata = {}
        return self._metadata

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._metadata = metadata

    @staticmethod
    def _normalize_method(method: str) -> str:
        value = str(method or "faiss").strip().lower().replace("-", "_")
        aliases = {
            "kd": "kd_tree",
            "kdtree": "kd_tree",
            "bruteforce": "brute",
            "brute_force": "brute",
        }
        value = aliases.get(value, value)
        return value if value in {"faiss", "brute", "hnsw", "pq", "kd_tree"} else "faiss"

    @staticmethod
    def _indexer_for(method: str):
        factories = {
            "faiss": FaissFlatIPIndex,
            "brute": BruteForceIndex,
            "hnsw": HNSWIndex,
            "pq": PQIndex,
            "kd_tree": KDTreeIndex,
        }
        return factories[method]()

    @staticmethod
    def _method_label(method: str) -> str:
        return {
            "faiss": "FlatIP",
            "brute": "BruteForce",
            "hnsw": "HNSW",
            "pq": "PQ",
            "kd_tree": "KD-Tree",
        }.get(method, "FlatIP")
