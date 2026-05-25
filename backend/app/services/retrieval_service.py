from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

from backend.app.database import execute, fetch_all, fetch_one
from backend.app.services.feature_service import get_feature_service
from backend.app.services.gallery_service import GalleryService
from backend.app.services.vector_index_service import VectorIndexService
from backend.app.utils.attribute_schema import (
    attribute_options,
    image_attributes,
    matches_attribute_values,
    selected_attribute_labels,
)
from backend.app.utils.file_utils import resolve_project_path
from backend.app.utils.image_utils import save_query_image, save_query_image_from_url, thumbnail_to_data_url
from backend.app.utils.metric_utils import map_at_k, normalize_vector, recall_at_k


SEARCH_MODES = {"feature", "filter", "hybrid"}


class RetrievalService:
    def __init__(self):
        self.gallery_service = GalleryService()
        self.vector_index_service = VectorIndexService()

    def search_by_gallery_image(
        self,
        image_id: int,
        method: str,
        top_k: int,
        user_id: int | None,
        feature_type: str = "resnet101",
        rerank_enabled: bool = False,
    ) -> dict:
        query_row = fetch_one("SELECT * FROM images WHERE id = %s AND is_deleted = 0 LIMIT 1", (image_id,))
        if not query_row:
            raise ValueError("查询图片不存在")
        requested_feature_type = self._normalize_feature_type(feature_type)
        if requested_feature_type == "none":
            query_vector = np.load(resolve_project_path(query_row["feature_path"])).astype(np.float32)
            effective_feature_type = self._feature_type_from_model(query_row.get("feature_model"))
        else:
            query_vector, effective_feature_type = self._extract_query_vector(query_row["file_path"], requested_feature_type)
        query_thumbnail = thumbnail_to_data_url(query_row["thumbnail_path"])
        return self._search(
            query_vector=query_vector,
            query_image_id=image_id,
            query_name=query_row["original_name"],
            query_source="gallery",
            query_label=query_row["label_name"],
            query_thumbnail=query_thumbnail,
            method=method,
            top_k=top_k,
            user_id=user_id,
            excluded_image_id=image_id,
            feature_type=feature_type,
            effective_feature_type=effective_feature_type,
            rerank_enabled=rerank_enabled,
        )

    def search_by_upload(
        self,
        data_url: str,
        original_name: str,
        method: str,
        top_k: int,
        user_id: int | None,
        feature_type: str = "resnet101",
        rerank_enabled: bool = False,
    ) -> dict:
        saved = save_query_image(data_url, original_name)
        query_vector, effective_feature_type = self._extract_query_vector(saved["file_path"], feature_type)
        return self._search(
            query_vector=query_vector,
            query_image_id=None,
            query_name=original_name,
            query_source="upload",
            query_label=None,
            query_thumbnail=thumbnail_to_data_url(saved["thumbnail_path"]),
            method=method,
            top_k=top_k,
            user_id=user_id,
            excluded_image_id=None,
            feature_type=feature_type,
            effective_feature_type=effective_feature_type,
            rerank_enabled=rerank_enabled,
        )

    def search_by_url(
        self,
        image_url: str,
        method: str,
        top_k: int,
        user_id: int | None,
        feature_type: str = "resnet101",
        rerank_enabled: bool = False,
    ) -> dict:
        saved = save_query_image_from_url(image_url)
        query_vector, effective_feature_type = self._extract_query_vector(saved["file_path"], feature_type)
        return self._search(
            query_vector=query_vector,
            query_image_id=None,
            query_name=image_url,
            query_source="url",
            query_label=None,
            query_thumbnail=thumbnail_to_data_url(saved["thumbnail_path"]),
            method=method,
            top_k=top_k,
            user_id=user_id,
            excluded_image_id=None,
            feature_type=feature_type,
            effective_feature_type=effective_feature_type,
            rerank_enabled=rerank_enabled,
        )

    def recognize_by_upload(self, data_url: str, original_name: str) -> dict:
        saved = save_query_image(data_url, original_name)
        query_vector = get_feature_service().extract(saved["file_path"], mode="baseline")
        return self._recognize_query(
            query_vector=query_vector,
            query_name=original_name,
            query_source="upload",
            query_thumbnail=thumbnail_to_data_url(saved["thumbnail_path"]),
            image_path=saved["file_path"],
        )

    def recognize_by_url(self, image_url: str) -> dict:
        saved = save_query_image_from_url(image_url)
        query_vector = get_feature_service().extract(saved["file_path"], mode="baseline")
        return self._recognize_query(
            query_vector=query_vector,
            query_name=image_url,
            query_source="url",
            query_thumbnail=thumbnail_to_data_url(saved["thumbnail_path"]),
            image_path=saved["file_path"],
        )

    def search_by_attributes(self, attribute_values: list[str], top_k: int, user_id: int | None, search_mode: str = "hybrid") -> dict:
        search_mode = search_mode if search_mode in SEARCH_MODES else "hybrid"
        selected_values = {str(value).strip() for value in attribute_values if str(value).strip()}
        selected_attributes = selected_attribute_labels(list(selected_values))
        if not selected_values:
            raise ValueError("请选择检索属性")

        feature_rows = self.gallery_service.load_feature_rows(include_thumbnails=False)
        candidate_rows = [row for row in feature_rows if self._row_matches(row, selected_values)]
        if not candidate_rows:
            raise ValueError("当前图库没有匹配属性")

        if search_mode == "filter":
            result = self._attribute_filter_result(candidate_rows, selected_attributes, top_k, user_id)
            result["metrics"]["searchMode"] = search_mode
            return result

        vectors = [np.asarray(row["feature"], dtype=np.float32).reshape(-1) for row in candidate_rows]
        query_vector = normalize_vector(np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32))
        query_name = "、".join(item["label"] for item in selected_attributes)
        expanded_k = min(len(feature_rows), max(top_k, top_k * 50))
        result = self._search(
            query_vector=query_vector,
            query_image_id=None,
            query_name=query_name or "属性检索",
            query_source="attributes",
            query_label=None,
            query_thumbnail="",
            method="faiss",
            top_k=expanded_k,
            user_id=user_id,
            excluded_image_id=None,
            attribute_values=selected_values,
            query_attributes=selected_attributes,
            search_k=expanded_k,
        )
        if search_mode == "hybrid":
            result["results"] = [item for item in result["results"] if item.get("attributeMatched")][:top_k]
        else:
            result["results"] = result["results"][:top_k]
        result["metrics"]["attributeCount"] = len(selected_attributes)
        result["metrics"]["candidateCount"] = len(candidate_rows)
        result["metrics"]["searchMode"] = search_mode
        result["recognition"] = self._recognize(result["results"], None)
        return result

    def _search(
        self,
        query_vector: np.ndarray,
        query_image_id: int | None,
        query_name: str,
        query_source: str,
        query_label: str | None,
        query_thumbnail: str,
        method: str,
        top_k: int,
        user_id: int | None,
        excluded_image_id: int | None,
        attribute_values: set[str] | None = None,
        query_attributes: list[dict] | None = None,
        search_k: int | None = None,
        feature_type: str = "resnet101",
        effective_feature_type: str | None = None,
        rerank_enabled: bool = False,
    ) -> dict:
        method = self._normalize_method(method)
        effective_feature_type = effective_feature_type or self._stored_feature_type()

        start = time.perf_counter()
        matches, items, index_status = self.vector_index_service.search(
            query_vector,
            search_k or top_k,
            excluded_image_id,
            method=method,
            rerank=rerank_enabled,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        if not items:
            raise ValueError("图库为空")

        results = []
        for match in matches:
            item = match["item"]
            score = match["score"]
            results.append(
                {
                    "id": item["id"],
                    "originalName": item["originalName"],
                    "labelName": item["labelName"],
                    "thumbnail": thumbnail_to_data_url(item["thumbnailPath"]),
                    "score": round(float(score), 4),
                    "attributeMatched": matches_attribute_values(item.get("attributeValues"), attribute_values or set()),
                    "attributes": self._result_attributes(item),
                    "source": item["source"],
                }
            )

        total_relevant = 0
        if query_label:
            total_relevant = sum(1 for item in items if item["labelName"] == query_label)
            if excluded_image_id:
                total_relevant = max(total_relevant - 1, 0)
        retrieved_labels = [item["labelName"] for item in results]
        metrics = {
            "mapAtK": map_at_k(query_label, retrieved_labels, total_relevant),
            "recallAtK": recall_at_k(query_label, retrieved_labels, total_relevant),
            "elapsedMs": elapsed_ms,
            "indexSizeBytes": index_status.get("indexSizeBytes", 0),
            "vectorCount": index_status.get("vectorCount", 0),
            "dimension": index_status.get("dimension", 0),
            "featureType": feature_type,
            "effectiveFeatureType": effective_feature_type,
            "rerankEnabled": bool(rerank_enabled),
        }
        if attribute_values:
            matched_count = sum(1 for item in results if item.get("attributeMatched"))
            relevant_total = sum(1 for item in items if matches_attribute_values(item.get("attributeValues"), attribute_values))
            metrics["attributePrecision"] = round(matched_count / max(len(results), 1), 4)
            metrics["attributeRecall"] = round(matched_count / max(relevant_total, 1), 4)
        execute(
            """
            INSERT INTO retrieval_logs (
                query_image_id, query_name, query_source, index_type, top_k, elapsed_ms,
                rerank_enabled, metrics_json, result_ids_json, created_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                query_image_id,
                query_name,
                query_source,
                method,
                top_k,
                elapsed_ms,
                1 if rerank_enabled else 0,
                json.dumps(metrics),
                json.dumps([item["id"] for item in results]),
                user_id,
            ),
        )
        return {
            "query": {
                "id": query_image_id,
                "name": query_name,
                "source": query_source,
                "labelName": query_label,
                "thumbnail": query_thumbnail,
                "attributes": query_attributes or [],
            },
            "results": results,
            "metrics": metrics,
            "recognition": self._recognize(results, query_label),
        }

    def rebuild_index(self) -> dict:
        return self.vector_index_service.rebuild_gallery_index()

    def index_status(self) -> dict:
        return self.vector_index_service.status()

    def attribute_options(self) -> list[dict]:
        rows = fetch_all(
            f"""
            SELECT label_name, COUNT(*) AS total
            FROM images
            WHERE {GalleryService._active_gallery_filter()} AND label_name IS NOT NULL AND label_name <> ''
            GROUP BY label_name
            ORDER BY label_name ASC
            """
        )
        counts = {row["label_name"]: int(row["total"]) for row in rows}
        return attribute_options(counts)

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

    def _extract_query_vector(self, image_path: str, feature_type: str) -> tuple[np.ndarray, str]:
        requested = self._normalize_feature_type(feature_type)
        if requested == "none":
            requested = self._stored_feature_type()
        mode = "embedding" if requested == "self_similarity" else "baseline"
        vector = get_feature_service().extract(image_path, mode=mode)
        gallery_dim = self._gallery_feature_dim()
        if gallery_dim and int(vector.shape[0]) != gallery_dim:
            fallback_type = self._stored_feature_type()
            fallback_mode = "embedding" if fallback_type == "self_similarity" else "baseline"
            fallback_vector = get_feature_service().extract(image_path, mode=fallback_mode)
            if int(fallback_vector.shape[0]) == gallery_dim:
                return fallback_vector, fallback_type
        return vector, requested if mode == "embedding" else "resnet101"

    @staticmethod
    def _normalize_feature_type(feature_type: str | None) -> str:
        value = str(feature_type or "none").strip().lower().replace("-", "_")
        aliases = {
            "": "none",
            "current": "none",
            "default": "none",
            "baseline": "resnet101",
            "resnet": "resnet101",
            "embedding": "self_similarity",
            "model": "self_similarity",
        }
        value = aliases.get(value, value)
        return value if value in {"none", "resnet101", "self_similarity"} else "none"

    @staticmethod
    def _feature_type_from_model(feature_model: str | None) -> str:
        model = str(feature_model or "")
        return "self_similarity" if "self" in model or "embedding" in model else "resnet101"

    @staticmethod
    def _gallery_feature_dim() -> int | None:
        row = fetch_one(
            f"""
            SELECT feature_dim, COUNT(*) AS total
            FROM images
            WHERE {GalleryService._active_gallery_filter()} AND feature_dim IS NOT NULL
            GROUP BY feature_dim
            ORDER BY total DESC
            LIMIT 1
            """
        )
        return int(row["feature_dim"]) if row else None

    @staticmethod
    def _stored_feature_type() -> str:
        row = fetch_one(
            f"""
            SELECT feature_model
            FROM images
            WHERE {GalleryService._active_gallery_filter()} AND feature_model IS NOT NULL
            GROUP BY feature_model
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """
        )
        model = str((row or {}).get("feature_model") or "")
        return RetrievalService._feature_type_from_model(model)

    @staticmethod
    def _row_matches(row: dict, selected_values: set[str]) -> bool:
        attrs = image_attributes(row.get("labelName"))
        return matches_attribute_values(attrs.attribute_values, selected_values)

    @staticmethod
    def _result_attributes(item: dict) -> list[dict[str, str]]:
        if item.get("attributes"):
            return item["attributes"]
        return image_attributes(item.get("labelName")).display_tags

    def _attribute_filter_result(self, candidate_rows: list[dict], selected_attributes: list[dict], top_k: int, user_id: int | None) -> dict:
        started_at = time.perf_counter()
        selected_rows = sorted(candidate_rows, key=lambda row: int(row["id"]), reverse=True)[:top_k]
        results = [
            {
                "id": int(row["id"]),
                "originalName": row["originalName"],
                "labelName": row["labelName"],
                "thumbnail": thumbnail_to_data_url(row["thumbnailPath"]),
                "score": 1.0,
                "attributeMatched": True,
                "attributes": image_attributes(row.get("labelName")).display_tags,
                "source": row["source"],
            }
            for row in selected_rows
        ]
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        metrics = {
            "mapAtK": None,
            "recallAtK": None,
            "elapsedMs": elapsed_ms,
            "indexSizeBytes": self.vector_index_service.status().get("indexSizeBytes", 0),
            "vectorCount": self.vector_index_service.status().get("vectorCount", 0),
            "dimension": self.vector_index_service.status().get("dimension", 0),
            "attributePrecision": 1.0,
            "attributeRecall": round(len(results) / max(len(candidate_rows), 1), 4),
            "candidateCount": len(candidate_rows),
            "attributeCount": len(selected_attributes),
        }
        query_name = "、".join(item["label"] for item in selected_attributes)
        execute(
            """
            INSERT INTO retrieval_logs (
                query_image_id, query_name, query_source, index_type, top_k, elapsed_ms,
                rerank_enabled, metrics_json, result_ids_json, created_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                None,
                query_name or "属性筛选",
                "attributes",
                "attribute_filter",
                top_k,
                elapsed_ms,
                0,
                json.dumps(metrics),
                json.dumps([item["id"] for item in results]),
                user_id,
            ),
        )
        return {
            "query": {
                "id": None,
                "name": query_name or "属性筛选",
                "source": "attributes",
                "labelName": None,
                "thumbnail": "",
                "attributes": selected_attributes,
            },
            "results": results,
            "metrics": metrics,
            "recognition": self._recognize(results, None),
        }

    def _recognize_query(self, query_vector: np.ndarray, query_name: str, query_source: str, query_thumbnail: str, image_path: str) -> dict:
        started_at = time.perf_counter()
        matches, _, index_status = self.vector_index_service.search(query_vector, 20)
        evidence = []
        for match in matches:
            item = match["item"]
            evidence.append(
                {
                    "id": item["id"],
                    "originalName": item["originalName"],
                    "labelName": item["labelName"],
                    "thumbnail": thumbnail_to_data_url(item["thumbnailPath"]),
                    "score": round(float(match["score"]), 4),
                    "attributes": self._result_attributes(item),
                }
            )
        recognition = self._recognize(evidence, None)
        predicted_label = recognition.get("predictedLabel")
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        feature = self._feature_profile(query_vector, image_path)
        class_comparison = self._class_feature_comparison(query_vector, evidence)
        final_label = class_comparison.get("primaryLabel") or predicted_label
        attrs = image_attributes(final_label if final_label != "未识别" else None)
        return {
            "query": {
                "name": query_name,
                "source": query_source,
                "thumbnail": query_thumbnail,
            },
            "predictedLabel": final_label,
            "confidence": class_comparison.get("decisionConfidence") or recognition.get("confidence"),
            "attributes": attrs.display_tags if final_label != "未识别" else [],
            "feature": feature,
            "classComparison": class_comparison,
            "candidates": recognition.get("candidates") or [],
            "evidence": evidence[:8],
            "metrics": {
                "elapsedMs": elapsed_ms,
                "vectorCount": index_status.get("vectorCount", 0),
                "dimension": index_status.get("dimension", 0),
            },
        }

    @staticmethod
    def _feature_profile(vector: np.ndarray, image_path: str) -> dict:
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return {
                "model": "resnet101_imagenet_pooling",
                "featureType": "深度图像嵌入",
                "dimension": 0,
                "visual": {},
                "embedding": {},
                "stats": {},
            }
        stats = {
            "mean": round(float(np.mean(flat)), 6),
            "std": round(float(np.std(flat)), 6),
            "min": round(float(np.min(flat)), 6),
            "max": round(float(np.max(flat)), 6),
            "l2Norm": round(float(np.linalg.norm(flat)), 6),
            "nonZero": int(np.count_nonzero(flat)),
        }
        return {
            "model": "resnet101_imagenet_pooling",
            "featureType": "深度图像嵌入",
            "dimension": int(flat.size),
            "visual": RetrievalService._visual_feature_summary(image_path),
            "embedding": {
                "stats": stats,
            },
            "stats": stats,
        }

    def _class_feature_comparison(self, query_vector: np.ndarray, evidence: list[dict]) -> dict:
        query = normalize_vector(np.asarray(query_vector, dtype=np.float32).reshape(-1))
        feature_rows = self.gallery_service.load_feature_rows(include_thumbnails=False)
        class_vectors: dict[str, list[np.ndarray]] = {}
        for row in feature_rows:
            label = row.get("labelName")
            if not label:
                continue
            vector = np.asarray(row.get("feature"), dtype=np.float32).reshape(-1)
            if vector.shape != query.shape:
                continue
            class_vectors.setdefault(label, []).append(normalize_vector(vector))

        class_payload = []
        centroids = {}
        for label, vectors in class_vectors.items():
            matrix = np.stack(vectors, axis=0)
            centroid = normalize_vector(np.mean(matrix, axis=0).astype(np.float32))
            scores = matrix @ query
            top_scores = np.sort(scores)[-min(5, scores.shape[0]) :]
            top_evidence_count = int(sum(1 for item in evidence if item.get("labelName") == label))
            top_evidence_ratio = top_evidence_count / max(len(evidence), 1)
            prototype_similarity = float(np.dot(query, centroid))
            nearest_similarity = float(np.max(scores))
            top_average_similarity = float(np.mean(top_scores))
            decision_score = (top_average_similarity * 0.45) + (nearest_similarity * 0.35) + (top_evidence_ratio * 0.2)
            centroids[label] = centroid
            class_payload.append(
                {
                    "label": label,
                    "sampleCount": int(matrix.shape[0]),
                    "decisionScore": round(float(decision_score), 4),
                    "prototypeSimilarity": round(prototype_similarity, 4),
                    "nearestSimilarity": round(nearest_similarity, 4),
                    "topAverageSimilarity": round(top_average_similarity, 4),
                    "topEvidenceCount": top_evidence_count,
                    "topEvidenceRatio": round(float(top_evidence_ratio), 4),
                }
            )

        class_payload.sort(key=lambda item: (item["decisionScore"], item["topEvidenceCount"], item["nearestSimilarity"]), reverse=True)
        top_classes = class_payload[:6]
        primary = class_payload[0] if class_payload else None
        runner = next((item for item in class_payload if primary and item["label"] != primary["label"]), None)
        prototype_margin = None
        nearest_margin = None
        decision_margin = None
        feature_groups = []
        if primary and runner:
            decision_margin = round(float(primary["decisionScore"] - runner["decisionScore"]), 4)
            prototype_margin = round(float(primary["prototypeSimilarity"] - runner["prototypeSimilarity"]), 4)
            nearest_margin = round(float(primary["nearestSimilarity"] - runner["nearestSimilarity"]), 4)
            feature_groups = self._discriminative_feature_groups(query, centroids[primary["label"]], centroids[runner["label"]], primary["label"], runner["label"])

        return {
            "totalImages": int(sum(len(vectors) for vectors in class_vectors.values())),
            "classCount": int(len(class_vectors)),
            "classes": top_classes,
            "primaryLabel": primary["label"] if primary else None,
            "runnerUpLabel": runner["label"] if runner else None,
            "decisionConfidence": primary["decisionScore"] if primary else None,
            "decisionMargin": decision_margin,
            "prototypeMargin": prototype_margin,
            "nearestMargin": nearest_margin,
            "topEvidenceTotal": len(evidence),
            "featureGroups": feature_groups,
        }

    @staticmethod
    def _discriminative_feature_groups(
        query: np.ndarray,
        primary_centroid: np.ndarray,
        runner_centroid: np.ndarray,
        primary_label: str,
        runner_label: str,
        group_count: int = 16,
    ) -> list[dict]:
        groups = []
        for index, indices in enumerate(np.array_split(np.arange(query.shape[0]), group_count), start=1):
            if indices.size == 0:
                continue
            primary_score = float(np.dot(query[indices], primary_centroid[indices]))
            runner_score = float(np.dot(query[indices], runner_centroid[indices]))
            difference = primary_score - runner_score
            groups.append(
                {
                    "name": f"嵌入特征组{index}",
                    "primaryScore": round(primary_score, 4),
                    "runnerScore": round(runner_score, 4),
                    "difference": round(difference, 4),
                    "supportLabel": primary_label if difference >= 0 else runner_label,
                }
            )
        groups.sort(key=lambda item: abs(float(item["difference"])), reverse=True)
        return groups[:6]

    @staticmethod
    def _visual_feature_summary(image_path: str) -> dict:
        image = Image.open(resolve_project_path(image_path)).convert("RGB")
        width, height = image.size
        analysis_image = image.resize((224, 224))
        arr = np.asarray(analysis_image, dtype=np.float32) / 255.0
        gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        rgb_max = arr.max(axis=2)
        rgb_min = arr.min(axis=2)
        saturation = np.where(rgb_max > 1e-6, (rgb_max - rgb_min) / np.maximum(rgb_max, 1e-6), 0)
        rg = arr[..., 0] - arr[..., 1]
        yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
        colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        gradient = np.zeros_like(gray)
        gradient[:, 1:] += gx
        gradient[1:, :] += gy
        laplace = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        sharpness = min(float(np.var(laplace) * 80.0), 1.0)
        metrics = {
            "brightness": round(float(np.mean(gray)), 4),
            "contrast": round(float(np.std(gray) * 2.5), 4),
            "saturation": round(float(np.mean(saturation)), 4),
            "colorfulness": round(float(min(colorfulness * 1.8, 1.0)), 4),
            "edgeDensity": round(float(np.mean(gradient > 0.18)), 4),
            "textureStrength": round(float(min(np.mean(gradient) * 3.0, 1.0)), 4),
            "sharpness": round(sharpness, 4),
        }
        metrics = {key: max(0.0, min(float(value), 1.0)) for key, value in metrics.items()}
        return {
            "image": {
                "width": width,
                "height": height,
                "aspectRatio": round(width / max(height, 1), 4),
            },
            "dominantColors": RetrievalService._dominant_colors(image),
            "metrics": metrics,
            "descriptors": RetrievalService._visual_descriptors(metrics),
        }

    @staticmethod
    def _dominant_colors(image: Image.Image) -> list[dict]:
        sample = image.resize((96, 96))
        quantized = sample.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette() or []
        counts = quantized.getcolors(96 * 96) or []
        total = sum(count for count, _ in counts) or 1
        colors = []
        for count, palette_index in sorted(counts, reverse=True)[:5]:
            offset = int(palette_index) * 3
            if offset + 2 >= len(palette):
                continue
            red, green, blue = palette[offset : offset + 3]
            colors.append(
                {
                    "hex": f"#{red:02x}{green:02x}{blue:02x}",
                    "ratio": round(count / total, 4),
                }
            )
        return colors

    @staticmethod
    def _visual_descriptors(metrics: dict[str, float]) -> list[dict[str, str]]:
        def level(value: float, low: float, high: float, low_text: str, mid_text: str, high_text: str) -> str:
            if value < low:
                return low_text
            if value > high:
                return high_text
            return mid_text

        return [
            {
                "name": "亮度",
                "value": level(metrics["brightness"], 0.36, 0.64, "偏暗", "均衡", "明亮"),
            },
            {
                "name": "对比度",
                "value": level(metrics["contrast"], 0.18, 0.42, "柔和", "适中", "明显"),
            },
            {
                "name": "色彩",
                "value": level(metrics["colorfulness"], 0.16, 0.38, "克制", "自然", "丰富"),
            },
            {
                "name": "轮廓",
                "value": level(metrics["edgeDensity"], 0.08, 0.22, "简洁", "清晰", "密集"),
            },
            {
                "name": "纹理",
                "value": level(metrics["textureStrength"], 0.12, 0.28, "平滑", "适中", "细节明显"),
            },
            {
                "name": "清晰度",
                "value": level(metrics["sharpness"], 0.12, 0.36, "偏低", "正常", "较高"),
            },
        ]

    @staticmethod
    def _recognize(results: list[dict], query_label: str | None) -> dict:
        label_scores: dict[str, float] = {}
        label_counts: dict[str, int] = {}
        for item in results:
            label = item.get("labelName")
            if not label:
                continue
            score = max(float(item.get("score") or 0), 0.0)
            label_scores[label] = label_scores.get(label, 0.0) + score
            label_counts[label] = label_counts.get(label, 0) + 1

        if not label_scores:
            return {
                "predictedLabel": "未识别",
                "confidence": None,
                "candidates": [],
                "basisCount": 0,
                "queryLabel": query_label,
            }

        total_score = sum(label_scores.values()) or 1.0
        candidates = [
            {
                "label": label,
                "score": round(score, 4),
                "confidence": round(score / total_score, 4),
                "count": label_counts.get(label, 0),
            }
            for label, score in sorted(label_scores.items(), key=lambda entry: entry[1], reverse=True)
        ]
        best = candidates[0]
        return {
            "predictedLabel": best["label"],
            "confidence": best["confidence"],
            "candidates": candidates[:3],
            "basisCount": sum(label_counts.values()),
            "queryLabel": query_label,
        }
