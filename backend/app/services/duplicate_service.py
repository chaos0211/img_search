from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageEnhance

from backend.app.database import execute
from backend.app.services.feature_service import get_feature_service
from backend.app.services.gallery_service import GalleryService
from backend.app.utils.file_utils import resolve_project_path
from backend.app.utils.metric_utils import normalize_matrix


class DuplicateService:
    def __init__(self):
        self.gallery_service = GalleryService()

    def scan(self, threshold: float) -> list[dict]:
        items = self.gallery_service.load_feature_rows(include_thumbnails=False)
        if len(items) < 2:
            return []

        normalized_vectors = normalize_matrix(np.stack([item["feature"] for item in items], axis=0))
        pairs = []
        total = len(items)
        block_size = 256
        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            similarity_block = normalized_vectors[start:end] @ normalized_vectors[start:].T
            row_indices, column_offsets = np.where(similarity_block >= threshold)
            for row_index, column_offset in zip(row_indices.tolist(), column_offsets.tolist()):
                left_index = start + int(row_index)
                right_index = start + int(column_offset)
                if right_index <= left_index:
                    continue
                left_item = items[left_index]
                right_item = items[right_index]
                pairs.append(
                    {
                        "left": {
                            "id": left_item["id"],
                            "name": left_item["originalName"],
                            "thumbnailPath": left_item["thumbnailPath"],
                        },
                        "right": {
                            "id": right_item["id"],
                            "name": right_item["originalName"],
                            "thumbnailPath": right_item["thumbnailPath"],
                        },
                        "similarity": round(float(similarity_block[row_index, column_offset]), 4),
                    }
                )
        pairs.sort(key=lambda item: item["similarity"], reverse=True)
        return pairs

    def evaluate_thresholds(
        self,
        thresholds: list[float] | None = None,
        top_k: int = 10,
        sample_size: int = 100,
    ) -> dict:
        items = self.gallery_service.load_feature_rows(include_thumbnails=False)
        if len(items) < 2:
            return self._empty_threshold_result(thresholds or [], top_k, len(items))

        threshold_values = thresholds or [round(value / 100, 2) for value in range(90, 100)]
        threshold_values = sorted({round(float(value), 3) for value in threshold_values if 0 < float(value) < 1})
        sample_size = max(1, min(int(sample_size), len(items)))

        vectors = normalize_matrix(np.stack([item["feature"] for item in items], axis=0).astype(np.float32))
        feature_model = items[0].get("featureModel") or ""
        feature_dim = items[0].get("featureDim") or int(vectors.shape[1])
        pairs = self._build_validation_pairs(items, vectors, sample_size)

        rows = []
        for threshold in threshold_values:
            rows.append(self._evaluate_pair_threshold(threshold=threshold, pairs=pairs))

        recommended = max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"], row["threshold"])) if rows else None
        positive_pair_count = sum(1 for pair in pairs if pair["isDuplicate"])
        negative_pair_count = sum(1 for pair in pairs if not pair["isDuplicate"])
        return {
            "galleryCount": len(items),
            "featureModel": feature_model,
            "featureDim": int(feature_dim),
            "sampleSize": sample_size,
            "positivePairCount": positive_pair_count,
            "negativePairCount": negative_pair_count,
            "validationPairCount": len(pairs),
            "recommendedThreshold": recommended["threshold"] if recommended else None,
            "rows": rows,
        }

    @staticmethod
    def _empty_threshold_result(thresholds: list[float], top_k: int, gallery_count: int) -> dict:
        return {
            "galleryCount": gallery_count,
            "featureModel": "",
            "featureDim": 0,
            "sampleSize": 0,
            "positivePairCount": 0,
            "negativePairCount": 0,
            "validationPairCount": 0,
            "recommendedThreshold": None,
            "rows": [
                {
                    "threshold": round(float(threshold), 3),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "falsePositiveCount": 0,
                    "falseNegativeCount": 0,
                }
                for threshold in thresholds
            ],
        }

    def _build_validation_pairs(self, items: list[dict], vectors: np.ndarray, sample_size: int) -> list[dict]:
        if not items:
            return []
        sample_indices = np.linspace(0, len(items) - 1, num=min(sample_size, len(items)), dtype=int).tolist()
        pairs = []
        positive_pairs = self._build_positive_pairs(items, vectors, sample_indices)
        pairs.extend(positive_pairs)
        for left_index in sample_indices[: len(positive_pairs)]:
            right_index = self._hard_negative_index(vectors, int(left_index))
            if right_index is None:
                continue
            pairs.append(
                {
                    "leftIndex": int(left_index),
                    "rightIndex": int(right_index),
                    "similarity": round(float(vectors[int(left_index)] @ vectors[int(right_index)]), 6),
                    "isDuplicate": False,
                }
            )
        return pairs

    def _build_positive_pairs(self, items: list[dict], vectors: np.ndarray, sample_indices: list[int]) -> list[dict]:
        feature_service = get_feature_service()
        feature_mode = self._feature_mode(items[0])
        pairs = []
        with tempfile.TemporaryDirectory(prefix="duplicate-threshold-") as temp_dir:
            temp_root = Path(temp_dir)
            for order, left_index in enumerate(sample_indices):
                item = items[int(left_index)]
                source_path = resolve_project_path(item["filePath"])
                if not source_path.exists():
                    continue
                augmented_path = temp_root / f"duplicate_{left_index}.jpg"
                try:
                    self._write_augmented_image(source_path, augmented_path, order)
                    augmented_feature = feature_service.extract(str(augmented_path), mode=feature_mode)
                except Exception:
                    continue
                if int(augmented_feature.shape[0]) != int(vectors.shape[1]):
                    continue
                pairs.append(
                    {
                        "leftIndex": int(left_index),
                        "rightIndex": int(left_index),
                        "similarity": round(float(vectors[int(left_index)] @ augmented_feature.astype(np.float32)), 6),
                        "isDuplicate": True,
                    }
                )
        return pairs

    @staticmethod
    def _feature_mode(item: dict) -> str:
        feature_model = str(item.get("featureModel") or "").lower()
        feature_dim = int(item.get("featureDim") or 0)
        if "self" in feature_model or "embedding" in feature_model or feature_dim == 1024:
            return "embedding"
        return "baseline"

    @staticmethod
    def _write_augmented_image(source_path: Path, target_path: Path, order: int) -> None:
        image = Image.open(source_path).convert("RGB")
        operations = order % 4
        if operations == 1:
            image = ImageEnhance.Brightness(image).enhance(1.01)
        elif operations == 2:
            image = ImageEnhance.Contrast(image).enhance(1.02)
        elif operations == 3:
            image = ImageEnhance.Color(image).enhance(0.98)
        image.save(target_path, format="JPEG", quality=98)

    @staticmethod
    def _hard_negative_index(vectors: np.ndarray, left_index: int) -> int | None:
        if vectors.shape[0] < 2:
            return None
        scores = vectors[int(left_index)] @ vectors.T
        scores[int(left_index)] = -np.inf
        right_index = int(np.argmax(scores))
        if not np.isfinite(scores[right_index]):
            return None
        return right_index

    @staticmethod
    def _evaluate_pair_threshold(threshold: float, pairs: list[dict]) -> dict:
        tp = fp = fn = tn = 0
        for pair in pairs:
            predicted_duplicate = float(pair["similarity"]) >= float(threshold)
            is_duplicate = bool(pair["isDuplicate"])
            if is_duplicate and predicted_duplicate:
                tp += 1
            elif not is_duplicate and predicted_duplicate:
                fp += 1
            elif is_duplicate and not predicted_duplicate:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "threshold": round(float(threshold), 3),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "falsePositiveCount": int(fp),
            "falseNegativeCount": int(fn),
        }

    def delete_duplicate(self, primary_image_id: int, duplicate_image_id: int, similarity: float, threshold: float, user_id: int | None):
        self.gallery_service.delete_image(duplicate_image_id)
        execute(
            """
            INSERT INTO duplicate_actions (
                primary_image_id, duplicate_image_id, similarity, threshold_value, action_type, acted_by
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (primary_image_id, duplicate_image_id, similarity, threshold, "delete", user_id),
        )
