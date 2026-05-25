from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.indexers.brute_force_index import BruteForceIndex
from backend.app.indexers.faiss_index import FaissFlatIPIndex
from backend.app.indexers.hnsw_index import HNSWIndex
from backend.app.indexers.kd_tree_index import KDTreeIndex
from backend.app.indexers.pq_index import PQIndex
from backend.app.services.rerank_service import RerankService
from backend.app.utils.file_utils import read_csv, resolve_project_path, serialize_project_path
from backend.app.utils.metric_utils import map_at_k, normalize_matrix, precision_at_k, recall_at_k


MATRIX_CACHE_VERSION = 2


class OfflineEvaluationService:
    def __init__(self):
        self.rerank_service = RerankService()
        self.index_factories = {
            "faiss": FaissFlatIPIndex,
            "brute": BruteForceIndex,
            "kd_tree": KDTreeIndex,
            "hnsw": HNSWIndex,
            "pq": PQIndex,
        }
        self.index_root = settings.index_root / "offline"
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.metric_root = settings.output_root / "metrics"
        self.metric_root.mkdir(parents=True, exist_ok=True)
        self.matrix_root = settings.project_root / "features" / "_matrix_cache"
        self.matrix_root.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        gallery_manifest_path: str,
        query_manifest_path: str,
        index_type: str,
        top_k: int = 10,
        rerank: bool = False,
        result_name: str = "evaluation.json",
        feature_scheme: str | None = None,
        feature_label: str | None = None,
        run_id: str | None = None,
        run_label: str | None = None,
        created_at: str | None = None,
        feature_extraction_ms: float = 0.0,
        matrix_load_ms: float = 0.0,
        database_ms: float = 0.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        gallery_data = self.load_feature_dataset_payload(gallery_manifest_path)
        query_data = self.load_feature_dataset_payload(query_manifest_path)

        return self.evaluate_loaded(
            gallery_vectors=gallery_data["vectors"],
            gallery_labels=gallery_data["labels"],
            query_vectors=query_data["vectors"],
            query_labels=query_data["labels"],
            gallery_ids=gallery_data["ids"],
            query_ids=query_data["ids"],
            gallery_id=gallery_data["datasetId"],
            query_set_id=query_data["datasetId"],
            index_type=index_type,
            top_k=top_k,
            rerank=rerank,
            result_name=result_name,
            feature_scheme=feature_scheme,
            feature_label=feature_label,
            run_id=run_id,
            run_label=run_label,
            created_at=created_at,
            feature_extraction_ms=feature_extraction_ms,
            matrix_load_ms=matrix_load_ms,
            database_ms=database_ms,
            feature_metadata=gallery_data.get("featureMetadata", {}),
            progress_callback=progress_callback,
        )

    def load_feature_dataset(
        self,
        manifest_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        payload = self.load_feature_dataset_payload(manifest_path, progress_callback=progress_callback)
        return payload["vectors"], payload["labels"]

    def load_feature_dataset_payload(
        self,
        manifest_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        manifest = resolve_project_path(manifest_path)
        frame = read_csv(manifest)
        if frame.empty or "feature_path" not in frame.columns:
            empty = np.empty((0, 0), dtype=np.float32)
            return {
                "vectors": empty,
                "labels": [],
                "ids": [],
                "datasetId": self._dataset_id(manifest, frame, []),
                "manifestPath": serialize_project_path(manifest),
                "featureMetadata": {},
            }

        labels = frame["label_name"].fillna("").astype(str).tolist()
        ids = self._dataset_ids(frame)
        feature_paths = frame["feature_path"].fillna("").astype(str).tolist()
        cache_key = self._matrix_cache_key(manifest)
        matrix_path = self.matrix_root / f"{cache_key}.npy"
        labels_path = self.matrix_root / f"{cache_key}_labels.json"
        ids_path = self.matrix_root / f"{cache_key}_ids.json"
        meta_path = self.matrix_root / f"{cache_key}_meta.json"
        expected_meta = self._matrix_meta(manifest, frame, feature_paths)

        if matrix_path.exists() and labels_path.exists() and ids_path.exists() and meta_path.exists():
            try:
                current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if current_meta == expected_meta:
                    vectors = np.load(matrix_path).astype(np.float32, copy=False)
                    cached_labels = json.loads(labels_path.read_text(encoding="utf-8"))
                    cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
                    if len(cached_labels) == len(vectors) and len(cached_ids) == len(vectors):
                        if progress_callback:
                            progress_callback(len(vectors), len(vectors))
                        return {
                            "vectors": vectors,
                            "labels": [str(item) for item in cached_labels],
                            "ids": [str(item) for item in cached_ids],
                            "datasetId": self._dataset_id(manifest, frame, feature_paths),
                            "manifestPath": serialize_project_path(manifest),
                            "featureMetadata": self._feature_metadata(frame),
                        }
            except Exception:
                pass

        vectors: list[np.ndarray] = []
        total = len(feature_paths)
        for index, feature_path in enumerate(feature_paths, start=1):
            vectors.append(np.load(resolve_project_path(feature_path)).astype(np.float32, copy=False).reshape(-1))
            if progress_callback and (index == total or index % 1000 == 0):
                progress_callback(index, total)

        matrix = np.stack(vectors, axis=0).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
        matrix = normalize_matrix(matrix)
        np.save(matrix_path, matrix)
        labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(expected_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "vectors": matrix,
            "labels": labels,
            "ids": ids,
            "datasetId": self._dataset_id(manifest, frame, feature_paths),
            "manifestPath": serialize_project_path(manifest),
            "featureMetadata": self._feature_metadata(frame),
        }

    def evaluate_loaded(
        self,
        *,
        gallery_vectors: np.ndarray,
        gallery_labels: list[str],
        query_vectors: np.ndarray,
        query_labels: list[str],
        gallery_ids: list[str] | None = None,
        query_ids: list[str] | None = None,
        gallery_id: str | None = None,
        query_set_id: str | None = None,
        index_type: str,
        top_k: int = 10,
        rerank: bool = False,
        result_name: str = "evaluation.json",
        feature_scheme: str | None = None,
        feature_label: str | None = None,
        run_id: str | None = None,
        run_label: str | None = None,
        created_at: str | None = None,
        feature_extraction_ms: float = 0.0,
        matrix_load_ms: float = 0.0,
        database_ms: float = 0.0,
        feature_metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if len(gallery_vectors) == 0 or len(query_vectors) == 0:
            raise ValueError("评估特征为空")
        gallery_vectors = normalize_matrix(np.asarray(gallery_vectors, dtype=np.float32))
        query_vectors = normalize_matrix(np.asarray(query_vectors, dtype=np.float32))
        gallery_ids = [str(item) for item in (gallery_ids or [f"gallery_{index}" for index in range(len(gallery_vectors))])]
        query_ids = [str(item) for item in (query_ids or [f"query_{index}" for index in range(len(query_vectors))])]
        if len(gallery_ids) != len(gallery_vectors) or len(query_ids) != len(query_vectors):
            raise ValueError("特征ID数量与向量数量不一致")

        if index_type not in self.index_factories:
            raise ValueError("unsupported index type")
        indexer = self.index_factories[index_type]()
        total_started_at = time.perf_counter()
        build_started_at = time.perf_counter()
        indexer.build(gallery_vectors)
        index_build_ms = (time.perf_counter() - build_started_at) * 1000

        index_path = self.index_root / f"{index_type}_{Path(result_name).stem}"
        if index_type == "faiss":
            index_path = index_path.with_suffix(".index")
        elif index_type == "hnsw":
            index_path = index_path.with_suffix(".bin")
        elif index_type == "pq":
            index_path = index_path.with_suffix(".npz")
        else:
            index_path = index_path.with_suffix(".npy")
        save_started_at = time.perf_counter()
        indexer.save(index_path)
        index_save_ms = (time.perf_counter() - save_started_at) * 1000
        storage_size = self._storage_breakdown(index_path, gallery_vectors, query_vectors)

        ap_scores = []
        recall_scores = []
        precision_scores = []
        search_elapsed_list = []
        rerank_elapsed_list = []
        total_queries = len(query_vectors)
        gallery_label_array = np.asarray(gallery_labels, dtype=object)
        relevant_counts = Counter(gallery_labels)
        gallery_id_to_label = {image_id: label for image_id, label in zip(gallery_ids, gallery_labels)}
        for query_index, (vector, query_label, query_image_id) in enumerate(zip(query_vectors, query_labels, query_ids), start=1):
            candidate_count = min(len(gallery_vectors), max(top_k, 20))
            search_started_at = time.perf_counter()
            indices, scores = indexer.search(vector, candidate_count)
            search_elapsed_list.append((time.perf_counter() - search_started_at) * 1000)
            if rerank:
                rerank_started_at = time.perf_counter()
                indices, scores = self.rerank_service.rerank(vector, gallery_vectors, indices, scores, candidate_count)
                rerank_elapsed_list.append((time.perf_counter() - rerank_started_at) * 1000)
            else:
                rerank_elapsed_list.append(0.0)
            indices, scores = self._remove_self_match(indices, scores, gallery_ids, query_image_id)
            top_indices = np.asarray(indices[:top_k], dtype=np.int64)
            retrieved_labels = gallery_label_array[top_indices].tolist()
            total_relevant = relevant_counts.get(query_label, 0)
            if gallery_id_to_label.get(query_image_id) == query_label:
                total_relevant = max(total_relevant - 1, 0)
            ap_scores.append(map_at_k(query_label, retrieved_labels, total_relevant) or 0.0)
            recall_scores.append(recall_at_k(query_label, retrieved_labels, total_relevant) or 0.0)
            precision_scores.append(precision_at_k(query_label, retrieved_labels, top_k) or 0.0)
            if progress_callback and (query_index == total_queries or query_index % 100 == 0):
                progress_callback(query_index, total_queries)

        search_ms = round(float(np.mean(search_elapsed_list)), 4) if search_elapsed_list else 0.0
        rerank_ms = round(float(np.mean(rerank_elapsed_list)), 4) if rerank_elapsed_list else 0.0
        total_evaluation_ms = round((time.perf_counter() - total_started_at) * 1000, 4)
        feature_type = feature_scheme or feature_label or "unknown"
        feature_metadata = feature_metadata or {}
        index_metadata = indexer.metadata()
        index_method = str(index_metadata.get("index_method") or index_metadata.get("index_class") or index_type)
        index_library = str(index_metadata.get("library") or "")
        result = {
            "runId": run_id,
            "runLabel": run_label,
            "createdAt": created_at,
            "featureScheme": feature_scheme,
            "featureLabel": feature_label,
            "featureType": feature_type,
            "feature_type": feature_type,
            "featureMetadata": feature_metadata,
            "featureModelName": feature_metadata.get("model_name"),
            "featureArchitecture": feature_metadata.get("architecture"),
            "checkpointPath": feature_metadata.get("checkpoint_path"),
            "checkpointMtimeNs": feature_metadata.get("checkpoint_mtime_ns"),
            "checkpointSizeBytes": feature_metadata.get("checkpoint_size_bytes"),
            "indexType": index_type,
            "index_type": index_type,
            "indexMethod": index_method,
            "index_method": index_method,
            "indexLibrary": index_library,
            "index_library": index_library,
            "topK": top_k,
            "top_k": top_k,
            "rerank": rerank,
            "rerankEnabled": rerank,
            "rerank_enabled": rerank,
            "galleryId": gallery_id or self._runtime_dataset_id("gallery", gallery_ids, gallery_labels),
            "gallery_id": gallery_id or self._runtime_dataset_id("gallery", gallery_ids, gallery_labels),
            "querySetId": query_set_id or self._runtime_dataset_id("query", query_ids, query_labels),
            "query_set_id": query_set_id or self._runtime_dataset_id("query", query_ids, query_labels),
            "metricType": "same_label_relevance",
            "metric_type": "same_label_relevance",
            "galleryCount": len(gallery_vectors),
            "queryCount": len(query_vectors),
            "mapAtK": round(float(np.mean(ap_scores)), 4) if ap_scores else 0.0,
            "recallAtK": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
            "precisionAtK": round(float(np.mean(precision_scores)), 4) if precision_scores else 0.0,
            "averageElapsedMs": round(search_ms + rerank_ms, 4),
            "indexSizeBytes": int(storage_size["indexFileSizeBytes"]),
            "timingMs": {
                "featureExtractionMs": round(float(feature_extraction_ms), 4),
                "matrixLoadMs": round(float(matrix_load_ms), 4),
                "indexBuildMs": round(index_build_ms, 4),
                "indexSaveMs": round(index_save_ms, 4),
                "indexSearchMs": search_ms,
                "rerankMs": rerank_ms,
                "databaseMs": round(float(database_ms), 4),
                "interfaceTotalMs": total_evaluation_ms,
                "totalEvaluationMs": total_evaluation_ms,
            },
            "storageSize": storage_size,
            "indexMetadata": index_metadata,
        }
        output_path = self.metric_root / result_name
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _matrix_cache_key(self, manifest_path: Path) -> str:
        relative_path = serialize_project_path(manifest_path)
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
        return f"{manifest_path.stem}_{digest}"

    def _matrix_meta(self, manifest_path: Path, frame: pd.DataFrame, feature_paths: list[str]) -> dict[str, Any]:
        first_feature = resolve_project_path(feature_paths[0]) if feature_paths else None
        last_feature = resolve_project_path(feature_paths[-1]) if feature_paths else None
        return {
            "cacheVersion": MATRIX_CACHE_VERSION,
            "manifestPath": serialize_project_path(manifest_path),
            "manifestMtimeNs": manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0,
            "rowCount": int(len(frame)),
            "firstFeaturePath": feature_paths[0] if feature_paths else "",
            "lastFeaturePath": feature_paths[-1] if feature_paths else "",
            "firstFeatureMtimeNs": first_feature.stat().st_mtime_ns if first_feature and first_feature.exists() else 0,
            "lastFeatureMtimeNs": last_feature.stat().st_mtime_ns if last_feature and last_feature.exists() else 0,
            "featureMode": self._frame_first_value(frame, "feature_mode"),
            "checkpointPath": self._frame_first_value(frame, "checkpoint_path"),
            "modelName": self._frame_first_value(frame, "model_name"),
            "architecture": self._frame_first_value(frame, "architecture"),
            "normalized": True,
            "dtype": "float32",
        }

    @staticmethod
    def _dataset_ids(frame: pd.DataFrame) -> list[str]:
        if "image_id" in frame.columns:
            return frame["image_id"].fillna("").astype(str).tolist()
        if "file_path" in frame.columns:
            return frame["file_path"].fillna("").astype(str).tolist()
        return [str(index) for index in range(len(frame))]

    @staticmethod
    def _frame_first_value(frame: pd.DataFrame, column: str) -> str:
        if column not in frame.columns:
            return ""
        values = frame[column].dropna().astype(str).unique().tolist()
        return values[0] if values else ""

    def _feature_metadata(self, frame: pd.DataFrame) -> dict[str, str]:
        return {
            key: self._frame_first_value(frame, key)
            for key in (
                "feature_mode",
                "checkpoint_path",
                "model_name",
                "architecture",
                "checkpoint_mtime_ns",
                "checkpoint_size_bytes",
            )
            if self._frame_first_value(frame, key)
        }

    def _dataset_id(self, manifest: Path, frame: pd.DataFrame, feature_paths: list[str]) -> str:
        payload = {
            "manifest": serialize_project_path(manifest),
            "rowCount": int(len(frame)),
            "featureMode": self._frame_first_value(frame, "feature_mode"),
            "checkpointPath": self._frame_first_value(frame, "checkpoint_path"),
            "modelName": self._frame_first_value(frame, "model_name"),
            "architecture": self._frame_first_value(frame, "architecture"),
            "firstFeaturePath": feature_paths[0] if feature_paths else "",
            "lastFeaturePath": feature_paths[-1] if feature_paths else "",
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        return f"{manifest.stem}_{digest}"

    @staticmethod
    def _runtime_dataset_id(prefix: str, image_ids: list[str], labels: list[str]) -> str:
        digest = hashlib.sha1(json.dumps([image_ids[:3], image_ids[-3:], labels[:3], labels[-3:], len(image_ids)], ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    @staticmethod
    def _remove_self_match(
        indices: np.ndarray,
        scores: np.ndarray,
        gallery_ids: list[str],
        query_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not query_id:
            return indices, scores
        filtered_indices = []
        filtered_scores = []
        for index, score in zip(indices.tolist(), scores.tolist()):
            if int(index) < 0 or int(index) >= len(gallery_ids):
                continue
            if gallery_ids[int(index)] == query_id:
                continue
            filtered_indices.append(int(index))
            filtered_scores.append(float(score))
        return np.asarray(filtered_indices, dtype=np.int64), np.asarray(filtered_scores, dtype=np.float32)

    @staticmethod
    def _storage_breakdown(index_path: Path, gallery_vectors: np.ndarray, query_vectors: np.ndarray) -> dict[str, int]:
        index_file_size = index_path.stat().st_size if index_path.exists() else 0
        metadata_size = 0
        sidecar = index_path.with_suffix(".json")
        if sidecar.exists():
            metadata_size += sidecar.stat().st_size
        feature_size = int(gallery_vectors.nbytes)
        query_feature_size = int(query_vectors.nbytes)
        return {
            "featureSizeBytes": feature_size,
            "queryFeatureSizeBytes": query_feature_size,
            "indexFileSizeBytes": int(index_file_size),
            "metadataSizeBytes": int(metadata_size),
            "totalStorageSizeBytes": int(feature_size + query_feature_size + index_file_size + metadata_size),
        }

    def scan_duplicate_thresholds(self, gallery_manifest_path: str, thresholds: list[float]) -> dict[str, Any]:
        gallery_df = read_csv(gallery_manifest_path)
        vectors = normalize_matrix(
            np.stack([np.load(resolve_project_path(path)).astype(np.float32) for path in gallery_df["feature_path"].tolist()], axis=0)
        )
        labels = np.asarray(gallery_df["label_name"].tolist())

        thresholds = sorted(float(threshold) for threshold in thresholds)
        stats = {
            threshold: {"tp": 0, "fp": 0, "fn": 0}
            for threshold in thresholds
        }
        block_size = 512
        total = len(labels)
        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            score_block = vectors[start:end] @ vectors[start:].T
            same_label_block = labels[start:end, None] == labels[None, start:]
            row_indices = np.arange(start, end)[:, None]
            column_indices = np.arange(start, total)[None, :]
            upper_mask = column_indices > row_indices
            same_label_block &= upper_mask

            for threshold in thresholds:
                predicted_block = (score_block >= threshold) & upper_mask
                tp = int(np.count_nonzero(predicted_block & same_label_block))
                fp = int(np.count_nonzero(predicted_block & ~same_label_block))
                fn = int(np.count_nonzero((~predicted_block) & same_label_block))
                stats[threshold]["tp"] += tp
                stats[threshold]["fp"] += fp
                stats[threshold]["fn"] += fn

        results = []
        for threshold in thresholds:
            tp = stats[threshold]["tp"]
            fp = stats[threshold]["fp"]
            fn = stats[threshold]["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            results.append(
                {
                    "threshold": threshold,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "predictedPairs": int(tp + fp),
                    "truePositivePairs": int(tp),
                }
            )
        output = {"thresholds": results}
        (self.metric_root / "duplicate_threshold_scan.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
