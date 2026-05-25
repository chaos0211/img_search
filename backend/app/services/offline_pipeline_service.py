from __future__ import annotations

import json
import multiprocessing
import os
import signal
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.app.config import settings
from backend.app.experiments.exp1_baseline import run as run_exp1
from backend.app.experiments.exp2_embedding import run as run_exp2
from backend.app.experiments.exp3_rerank import run as run_exp3
from backend.app.experiments.exp4_index_compare import run as run_exp4
from backend.app.experiments.exp5_duplicate_threshold import run as run_exp5
from backend.app.models.self_similarity_embedding import SelfSimilarityEmbedding
from backend.app.services.dataset_service import DatasetService
from backend.app.services.offline_evaluation_service import OfflineEvaluationService
from backend.app.services.training_service import EmbeddingTrainingService
from backend.app.utils.file_utils import read_csv, resolve_project_path, serialize_project_path


def _current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_runtime_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_runtime_file(path: Path, updates: dict[str, Any], replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {} if replace else _read_runtime_file(path)
    payload.update(updates)
    payload["updatedAt"] = _current_timestamp()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_offline_experiment_process(kwargs: dict[str, Any], runtime_path_text: str) -> None:
    runtime_path = Path(runtime_path_text)
    _write_runtime_file(
        runtime_path,
        {
            "isRunning": True,
            "status": "running",
            "progress": 0,
            "runId": kwargs.get("run_id"),
            "featureScheme": kwargs.get("feature_scheme") or "baseline",
            "rerankEnabled": bool(kwargs.get("rerank_enabled")),
            "pid": os.getpid(),
            "startedAt": kwargs.get("created_at") or _current_timestamp(),
            "finishedAt": None,
            "message": "",
        },
        replace=True,
    )

    def report(progress: int, message: str = "") -> None:
        _write_runtime_file(
            runtime_path,
            {
                "isRunning": True,
                "status": "running",
                "progress": max(0, min(99, int(progress))),
                "message": message,
            },
        )

    try:
        service = OfflinePipelineService()
        service.run_experiment(**kwargs, progress_callback=report)
        _write_runtime_file(
            runtime_path,
            {
                "isRunning": False,
                "status": "completed",
                "progress": 100,
                "finishedAt": _current_timestamp(),
                "message": "",
            },
        )
    except Exception as exc:
        runtime = _read_runtime_file(runtime_path)
        failed_progress = max(0, min(99, int(runtime.get("progress") or 0)))
        _write_runtime_file(
            runtime_path,
            {
                "isRunning": False,
                "status": "failed",
                "progress": failed_progress,
                "finishedAt": _current_timestamp(),
                "message": str(exc),
            },
        )


class OfflinePipelineService:
    MATRIX_INDEX_METHODS = ["brute", "kd_tree", "hnsw", "pq"]
    CIFAR10_CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self):
        self.project_root = settings.project_root
        self.dataset_service = DatasetService()
        self.training_service = EmbeddingTrainingService()
        self.metric_root = settings.output_root / "metrics"
        self.metric_root.mkdir(parents=True, exist_ok=True)
        self.experiment_runtime_path = self.metric_root / "offline_experiment_runtime.json"
        self._dataset_status_cache: tuple[tuple[tuple[str, int], ...], dict[str, Any]] | None = None
        self._experiment_runtime_lock = threading.Lock()
        self._experiment_runtime: dict[str, Any] = self._default_experiment_runtime()
        self._experiment_process: multiprocessing.Process | None = None

    def prepare_dataset(
        self,
        train_limit: int,
        test_limit: int,
        gallery_per_class: int,
        query_per_class: int,
        clean_existing: bool,
    ) -> dict[str, Any]:
        return self.dataset_service.prepare_cifar10(
            train_limit=train_limit,
            test_limit=test_limit,
            gallery_per_class=gallery_per_class,
            query_per_class=query_per_class,
            clean_existing=clean_existing,
        )

    def train_embedding(
        self,
        train_manifest_path: str,
        validation_manifest_path: str | None,
        epochs: int,
        early_stop_patience: int,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        optimizer_name: str,
        seed: int,
        save_best_only: bool,
        freeze_backbone: bool,
        device_name: str,
    ) -> dict[str, Any]:
        return self.training_service.train(
            train_manifest_path=str(self._resolve_path(train_manifest_path)),
            validation_manifest_path=str(self._resolve_path(validation_manifest_path)) if validation_manifest_path else None,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            seed=seed,
            save_best_only=save_best_only,
            freeze_backbone=freeze_backbone,
            device_name=device_name,
        )

    def start_training(
        self,
        train_manifest_path: str,
        validation_manifest_path: str | None,
        epochs: int,
        early_stop_patience: int,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        optimizer_name: str,
        seed: int,
        save_best_only: bool,
        freeze_backbone: bool,
        device_name: str,
    ) -> dict[str, Any]:
        if self.experiment_runtime_status().get("isRunning"):
            raise ValueError("评估任务进行中")
        return self.training_service.start_training(
            train_manifest_path=str(self._resolve_path(train_manifest_path)),
            validation_manifest_path=str(self._resolve_path(validation_manifest_path)) if validation_manifest_path else None,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            seed=seed,
            save_best_only=save_best_only,
            freeze_backbone=freeze_backbone,
            device_name=device_name,
        )

    def stop_training(self) -> dict[str, Any]:
        return self.training_service.stop_training()

    def extract_features(
        self,
        manifest_path: str,
        output_manifest_name: str,
        mode: str,
        checkpoint_path: str | None,
    ) -> dict[str, Any]:
        if mode != "embedding":
            raise ValueError("模型管理只导出训练模型特征")
        if not checkpoint_path:
            raise ValueError("请选择模型权重")
        checkpoint = None
        if checkpoint_path:
            checkpoint = str(self._resolve_path(checkpoint_path))
        model_name = self._current_model_name() if checkpoint else None
        return self.training_service.extract_features(
            manifest_path=str(self._resolve_path(manifest_path)),
            output_manifest_name=Path(output_manifest_name).name,
            checkpoint_path=checkpoint,
            mode=mode,
            model_name=model_name,
        )

    def extract_feature_set(self, checkpoint_path: str | None) -> dict[str, Any]:
        if not checkpoint_path:
            raise ValueError("请选择模型权重")
        checkpoint = str(self._resolve_path(checkpoint_path))
        model_name = self._current_model_name()
        gallery_result = self.training_service.extract_features(
            manifest_path=str(self.project_root / "data" / "manifests" / "cifar10_gallery.csv"),
            output_manifest_name="cifar10_gallery_embedding.csv",
            checkpoint_path=checkpoint,
            mode="embedding",
            model_name=model_name,
        )
        query_result = self.training_service.extract_features(
            manifest_path=str(self.project_root / "data" / "manifests" / "cifar10_query.csv"),
            output_manifest_name="cifar10_query_embedding.csv",
            checkpoint_path=checkpoint,
            mode="embedding",
            model_name=model_name,
        )
        return {"gallery": gallery_result, "query": query_result, "modelName": model_name}

    def run_experiment(
        self,
        experiment_name: str,
        top_k: int,
        gallery_manifest: str | None = None,
        query_manifest: str | None = None,
        baseline_gallery_manifest: str | None = None,
        baseline_query_manifest: str | None = None,
        embedding_gallery_manifest: str | None = None,
        embedding_query_manifest: str | None = None,
        feature_scheme: str | None = None,
        index_method: str | None = None,
        rerank_enabled: bool = False,
        run_id: str | None = None,
        created_at: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        if experiment_name == "matrix":
            return self._run_retrieval_matrix(
                top_k=top_k,
                baseline_gallery_manifest=baseline_gallery_manifest,
                baseline_query_manifest=baseline_query_manifest,
                embedding_gallery_manifest=embedding_gallery_manifest,
                embedding_query_manifest=embedding_query_manifest,
                feature_scheme=feature_scheme,
                index_method=index_method,
                rerank_enabled=rerank_enabled,
                run_id=run_id,
                created_at=created_at,
                progress_callback=progress_callback,
            )
        if experiment_name == "exp1":
            return run_exp1(str(self._resolve_required(gallery_manifest)), str(self._resolve_required(query_manifest)), top_k)
        if experiment_name == "exp2":
            return run_exp2(
                str(self._resolve_required(baseline_gallery_manifest)),
                str(self._resolve_required(baseline_query_manifest)),
                str(self._resolve_required(embedding_gallery_manifest)),
                str(self._resolve_required(embedding_query_manifest)),
                top_k,
            )
        if experiment_name == "exp3":
            return run_exp3(str(self._resolve_required(gallery_manifest)), str(self._resolve_required(query_manifest)), top_k)
        if experiment_name == "exp4":
            return run_exp4(str(self._resolve_required(gallery_manifest)), str(self._resolve_required(query_manifest)), top_k)
        if experiment_name == "exp5":
            return run_exp5(str(self._resolve_required(gallery_manifest)))
        raise ValueError("实验类型不存在")

    def start_experiment(
        self,
        experiment_name: str,
        top_k: int,
        gallery_manifest: str | None = None,
        query_manifest: str | None = None,
        baseline_gallery_manifest: str | None = None,
        baseline_query_manifest: str | None = None,
        embedding_gallery_manifest: str | None = None,
        embedding_query_manifest: str | None = None,
        feature_scheme: str | None = None,
        index_method: str | None = None,
        rerank_enabled: bool = False,
    ) -> dict[str, Any]:
        current_runtime = self.experiment_runtime_status()
        if current_runtime.get("isRunning"):
            raise ValueError("评估正在运行")
        if self.training_service.runtime_status().get("isRunning"):
            raise ValueError("训练任务进行中")
        self._validate_experiment_preflight(experiment_name, feature_scheme)

        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        started_at = self._current_timestamp()
        with self._experiment_runtime_lock:
            self._experiment_runtime = {
                "isRunning": True,
                "status": "running",
                "progress": 0,
                "runId": run_id,
                "featureScheme": feature_scheme or "baseline",
                "rerankEnabled": bool(rerank_enabled),
                "pid": None,
                "startedAt": started_at,
                "updatedAt": started_at,
                "finishedAt": None,
                "message": "",
            }

        kwargs = {
            "experiment_name": experiment_name,
            "top_k": top_k,
            "gallery_manifest": gallery_manifest,
            "query_manifest": query_manifest,
            "baseline_gallery_manifest": baseline_gallery_manifest,
            "baseline_query_manifest": baseline_query_manifest,
            "embedding_gallery_manifest": embedding_gallery_manifest,
            "embedding_query_manifest": embedding_query_manifest,
            "feature_scheme": feature_scheme,
            "index_method": index_method,
            "rerank_enabled": rerank_enabled,
            "run_id": run_id,
            "created_at": started_at,
        }
        _write_runtime_file(self.experiment_runtime_path, self._experiment_runtime, replace=True)
        process = multiprocessing.Process(
            target=_run_offline_experiment_process,
            args=(kwargs, str(self.experiment_runtime_path)),
            daemon=True,
            name="offline-evaluation",
        )
        process.start()
        self._experiment_process = process
        self._set_experiment_runtime(pid=process.pid)
        return self.experiment_runtime_status()

    def _validate_experiment_preflight(self, experiment_name: str, feature_scheme: str | None) -> None:
        if experiment_name != "matrix":
            return
        selected_feature = str(feature_scheme or "baseline")
        if selected_feature not in {"all", "none", "embedding", "self_similarity"}:
            return
        checkpoint_path = self.project_root / "checkpoints" / "embedding_best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = self.project_root / "checkpoints" / "embedding_latest.pt"
        if not checkpoint_path.exists():
            raise ValueError("缺少自相似特征模型权重")
        self.training_service.embedding_checkpoint_metadata(str(checkpoint_path))

    def experiment_runtime_status(self) -> dict[str, Any]:
        runtime = {**self._default_experiment_runtime(), **_read_runtime_file(self.experiment_runtime_path)}
        pid = runtime.get("pid")
        if not runtime.get("isRunning") and runtime.get("status") in {"cancelled", "idle"}:
            runtime["progress"] = 0
        if runtime.get("isRunning") and pid and not self._pid_alive(int(pid)):
            runtime.update(
                {
                    "isRunning": False,
                    "status": "failed",
                    "message": "评估进程已停止",
                    "finishedAt": self._current_timestamp(),
                }
            )
            _write_runtime_file(self.experiment_runtime_path, runtime)
        with self._experiment_runtime_lock:
            self._experiment_runtime = dict(runtime)
            return dict(runtime)

    def stop_experiment(self) -> dict[str, Any]:
        runtime = self.experiment_runtime_status()
        pid = runtime.get("pid")
        if pid and self._pid_alive(int(pid)):
            os.kill(int(pid), signal.SIGTERM)
            time.sleep(0.5)
            if self._pid_alive(int(pid)):
                os.kill(int(pid), signal.SIGKILL)
        runtime.update(
            {
                "isRunning": False,
                "status": "cancelled",
                "progress": 0,
                "finishedAt": self._current_timestamp(),
                "message": "",
            }
        )
        _write_runtime_file(self.experiment_runtime_path, runtime)
        with self._experiment_runtime_lock:
            self._experiment_runtime = dict(runtime)
        return runtime

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _run_retrieval_matrix(
        self,
        top_k: int,
        baseline_gallery_manifest: str | None,
        baseline_query_manifest: str | None,
        embedding_gallery_manifest: str | None,
        embedding_query_manifest: str | None,
        feature_scheme: str | None,
        index_method: str | None,
        rerank_enabled: bool,
        run_id: str | None = None,
        created_at: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        selected_feature = str(feature_scheme or "baseline")
        selected_index = str(index_method or "all")
        top_k = int(top_k or 10)
        run_id = run_id or datetime.now().strftime("%Y%m%d%H%M%S")
        created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feature_configs = []
        if selected_feature in {"all", "none", "baseline", "resnet101"}:
            feature_configs.append(
                {
                    "value": "baseline",
                    "label": "ResNet101",
                    "mode": "baseline",
                }
            )
        if selected_feature in {"all", "none", "embedding", "self_similarity"}:
            feature_configs.append(
                {
                    "value": "embedding",
                    "label": "自相似特征",
                    "mode": "embedding",
                }
            )
        if not feature_configs:
            raise ValueError("特征嵌入类型不存在")
        index_methods = list(self.MATRIX_INDEX_METHODS)
        if selected_index != "all":
            index_methods = [selected_index]

        service = OfflineEvaluationService()
        results: dict[str, Any] = {}
        total_units = max(1, len(feature_configs) * (4 + len(index_methods)))
        completed_units = 0

        def report(current_unit: int, current: int = 0, total: int = 1, label: str = "") -> None:
            if not progress_callback:
                return
            fraction = 1 if total <= 0 else max(0, min(1, current / total))
            percent = min(99, int(((current_unit + fraction) / total_units) * 100))
            progress_callback(percent, label)

        for feature in feature_configs:
            feature_started_at = time.perf_counter()
            gallery_manifest = self._ensure_offline_feature_manifest(
                "gallery",
                feature["mode"],
                progress_callback=lambda current, total, unit=completed_units: report(unit, current, total, "feature"),
            )
            completed_units += 1
            query_manifest = self._ensure_offline_feature_manifest(
                "query",
                feature["mode"],
                progress_callback=lambda current, total, unit=completed_units: report(unit, current, total, "feature"),
            )
            feature_extraction_ms = (time.perf_counter() - feature_started_at) * 1000
            completed_units += 1
            matrix_load_started_at = time.perf_counter()
            gallery_payload = service.load_feature_dataset_payload(
                gallery_manifest,
                progress_callback=lambda current, total, unit=completed_units: report(unit, current, total, "matrix"),
            )
            completed_units += 1
            query_payload = service.load_feature_dataset_payload(
                query_manifest,
                progress_callback=lambda current, total, unit=completed_units: report(unit, current, total, "matrix"),
            )
            matrix_load_ms = (time.perf_counter() - matrix_load_started_at) * 1000
            completed_units += 1
            rerank_label = "重排序" if rerank_enabled else "未重排序"
            feature_metadata = gallery_payload.get("featureMetadata", {})
            model_name = str(feature_metadata.get("model_name") or "").strip()
            model_label = f" {model_name}" if model_name and feature["value"] == "embedding" else ""
            run_label = f"{run_id} {feature['label']}{model_label} {rerank_label}"
            for index_type in index_methods:
                suffix = "_rerank" if rerank_enabled else ""
                key = f"{feature['value']}_{index_type}{suffix}"
                results[key] = service.evaluate_loaded(
                    gallery_vectors=gallery_payload["vectors"],
                    gallery_labels=gallery_payload["labels"],
                    gallery_ids=gallery_payload["ids"],
                    gallery_id=gallery_payload["datasetId"],
                    query_vectors=query_payload["vectors"],
                    query_labels=query_payload["labels"],
                    query_ids=query_payload["ids"],
                    query_set_id=query_payload["datasetId"],
                    index_type=index_type,
                    top_k=top_k,
                    rerank=rerank_enabled,
                    result_name=f"matrix_{run_id}_{feature['value']}_{index_type}{suffix}.json",
                    feature_scheme=feature["value"],
                    feature_label=feature["label"],
                    run_id=f"{run_id}_{feature['value']}_{'rerank' if rerank_enabled else 'plain'}",
                    run_label=run_label,
                    created_at=created_at,
                    feature_extraction_ms=feature_extraction_ms,
                    matrix_load_ms=matrix_load_ms,
                    feature_metadata=feature_metadata,
                    progress_callback=lambda current, total, unit=completed_units: report(unit, current, total, "evaluation"),
                )
                completed_units += 1
                report(completed_units, 0, 1, "evaluation")
        return results

    def _ensure_offline_feature_manifest(
        self,
        partition: str,
        mode: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        manifest_root = self.project_root / "data" / "manifests"
        source_manifest_map = {
            "gallery": manifest_root / "cifar10_train.csv",
            "query": manifest_root / "cifar10_test.csv",
        }
        source_manifest = source_manifest_map.get(partition, manifest_root / f"cifar10_{partition}.csv")
        output_manifest = self.project_root / "features" / f"cifar10_{partition}_{mode}.csv"
        expected_count = self._csv_row_count(source_manifest)

        checkpoint = None
        model_name = None
        expected_metadata: dict[str, str] = {"feature_mode": mode}
        if mode == "embedding":
            checkpoint_path = self.project_root / "checkpoints" / "embedding_best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = self.project_root / "checkpoints" / "embedding_latest.pt"
            if not checkpoint_path.exists():
                raise ValueError("缺少自相似特征模型权重")
            checkpoint = str(checkpoint_path)
            model_name = self._current_model_name()
            checkpoint_metadata = self.training_service.embedding_checkpoint_metadata(checkpoint)
            expected_metadata["checkpoint_path"] = serialize_project_path(checkpoint_path)
            expected_metadata["model_name"] = model_name
            expected_metadata["architecture"] = SelfSimilarityEmbedding.architecture_name
            expected_metadata["checkpoint_mtime_ns"] = str(checkpoint_metadata["checkpoint_mtime_ns"])
            expected_metadata["checkpoint_size_bytes"] = str(checkpoint_metadata["checkpoint_size_bytes"])

        if self._feature_manifest_complete(output_manifest, expected_count, expected_metadata):
            if progress_callback:
                progress_callback(expected_count, expected_count)
            return output_manifest

        resume = output_manifest.exists() and self._feature_manifest_has_metadata(output_manifest, expected_metadata)

        self.training_service.extract_features(
            manifest_path=str(source_manifest),
            output_manifest_name=output_manifest.name,
            checkpoint_path=checkpoint,
            mode=mode,
            model_name=model_name,
            resume=resume,
            progress_callback=progress_callback,
        )
        return output_manifest

    def _feature_manifest_complete(self, manifest_path: Path, expected_count: int, expected_metadata: dict[str, str] | None = None) -> bool:
        if expected_count <= 0 or not manifest_path.exists():
            return False
        frame = self._read_csv_optional(manifest_path)
        if frame is None or len(frame) != expected_count or "feature_path" not in frame.columns:
            return False
        if expected_metadata and not self._feature_frame_has_metadata(frame, expected_metadata):
            return False
        sample_paths = frame["feature_path"].dropna().astype(str).head(20).tolist()
        return bool(sample_paths) and all(resolve_project_path(path).exists() for path in sample_paths)

    def _feature_manifest_has_metadata(self, manifest_path: Path, expected_metadata: dict[str, str]) -> bool:
        frame = self._read_csv_optional(manifest_path)
        if frame is None or frame.empty:
            return False
        return self._feature_frame_has_metadata(frame, expected_metadata)

    @staticmethod
    def _feature_frame_has_metadata(frame: pd.DataFrame, expected_metadata: dict[str, str]) -> bool:
        for key, expected_value in expected_metadata.items():
            if key not in frame.columns:
                return False
            values = set(frame[key].dropna().astype(str).tolist())
            if values != {str(expected_value)}:
                return False
        return True

    def _set_experiment_progress(self, progress: int, message: str = "") -> None:
        self._set_experiment_runtime(progress=max(0, min(100, int(progress))), message=message)

    def _set_experiment_runtime(self, **updates: Any) -> None:
        with self._experiment_runtime_lock:
            self._experiment_runtime.update(updates)
            self._experiment_runtime["updatedAt"] = self._current_timestamp()
            _write_runtime_file(self.experiment_runtime_path, self._experiment_runtime)

    @staticmethod
    def _default_experiment_runtime() -> dict[str, Any]:
        return {
            "isRunning": False,
            "status": "idle",
            "progress": 0,
            "runId": None,
            "featureScheme": None,
            "rerankEnabled": False,
            "startedAt": None,
            "updatedAt": None,
            "finishedAt": None,
        }

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def dataset_status(self) -> dict[str, Any]:
        cache_key = self._dataset_cache_key()
        if self._dataset_status_cache and self._dataset_status_cache[0] == cache_key:
            return self._dataset_status_cache[1]

        summary_path = self.project_root / "data" / "manifests" / "cifar10_summary.json"
        summary = self._display_payload(self._read_json(summary_path, {}))
        raw_meta = self._raw_cifar10_metadata()
        raw_total_count = raw_meta["trainCount"] + raw_meta["testCount"]
        manifests = {
            name: self._read_csv_optional(self.project_root / "data" / "manifests" / name)
            for name in ("cifar10_train.csv", "cifar10_test.csv", "cifar10_gallery.csv", "cifar10_query.csv")
        }
        labels = raw_meta["classes"] or sorted(
            {
                label
                for dataframe in manifests.values()
                if dataframe is not None and not dataframe.empty
                for label in dataframe["label_name"].dropna().tolist()
            }
        )
        class_rows = []
        for label in labels:
            class_rows.append(
                {
                    "labelName": label,
                    "trainCount": self._count_label(manifests["cifar10_train.csv"], label),
                    "testCount": self._count_label(manifests["cifar10_test.csv"], label),
                    "galleryCount": self._count_label(manifests["cifar10_gallery.csv"], label),
                    "queryCount": self._count_label(manifests["cifar10_query.csv"], label),
                }
            )

        source_manifests = [manifests["cifar10_train.csv"], manifests["cifar10_test.csv"]]
        sha1_values: list[str] = []
        valid_count = 0
        for dataframe in source_manifests:
            if dataframe is None or dataframe.empty:
                continue
            valid_count += int(dataframe.get("is_valid", pd.Series(dtype=int)).sum()) if "is_valid" in dataframe.columns else len(dataframe)
            if "sha1" in dataframe.columns:
                sha1_values.extend(dataframe["sha1"].dropna().astype(str).tolist())
        duplicate_count = max(len(sha1_values) - len(set(sha1_values)), 0)
        prepared_train_count = int(summary.get("trainCount", 0))
        prepared_test_count = int(summary.get("testCount", 0))
        prepared_gallery_count = int(summary.get("galleryCount", 0))
        prepared_query_count = int(summary.get("queryCount", 0))
        summary_updated_at = self._format_optional_timestamp(summary_path)

        payload = {
            "selector": {
                "current": "cifar10",
                "options": [{"label": "CIFAR-10", "value": "cifar10"}],
            },
            "summary": {
                "rawTrainCount": raw_meta["trainCount"],
                "rawTestCount": raw_meta["testCount"],
                "rawTotalCount": raw_total_count,
                "preparedTrainCount": prepared_train_count,
                "preparedTestCount": prepared_test_count,
                "preparedTotalCount": prepared_train_count + prepared_test_count,
                "preparedGalleryCount": prepared_gallery_count,
                "preparedQueryCount": prepared_query_count,
                "validCount": valid_count,
                "duplicateCount": duplicate_count,
                "classCount": len(labels),
                "downloaded": raw_meta["downloaded"],
                "summaryUpdatedAt": summary_updated_at,
            },
            "basicInfo": [
                {"label": "数据集", "value": "CIFAR-10"},
                {"label": "来源", "value": "torchvision.datasets.CIFAR10"},
                {"label": "图像规格", "value": "32 x 32 / RGB"},
                {"label": "类别数", "value": len(labels)},
                {"label": "原始训练", "value": raw_meta["trainCount"]},
                {"label": "原始测试", "value": raw_meta["testCount"]},
                {"label": "原始总量", "value": raw_total_count},
                {"label": "数据状态", "value": "已下载" if raw_meta["downloaded"] else "未下载"},
                {"label": "原始目录", "value": "data/raw/cifar10"},
                {"label": "处理目录", "value": "data/processed/cifar10"},
                {"label": "图库目录", "value": "data/gallery/cifar10"},
                {"label": "查询目录", "value": "data/query/cifar10"},
            ],
            "preprocessScheme": [
                {"label": "数据获取", "value": "按 train / test 两个 split 下载 CIFAR-10 原始批文件"},
                {"label": "格式统一", "value": "逐张读取为 PIL Image，统一转换为 RGB PNG"},
                {"label": "有效性标记", "value": "图片成功导出后记录 is_valid = 1"},
                {"label": "指纹计算", "value": "对导出的 PNG 计算 SHA1"},
                {"label": "训练集导出", "value": "生成 data/processed/cifar10/train 与 cifar10_train.csv"},
                {"label": "测试集导出", "value": "生成 data/processed/cifar10/test 与 cifar10_test.csv"},
                {"label": "图库构建", "value": "从 train 按类别顺序截取前 n 张复制到 gallery"},
                {"label": "查询构建", "value": "从 test 按类别顺序截取前 n 张复制到 query"},
                {"label": "输出清单", "value": "同步生成 train / test / gallery / query 四份 CSV"},
            ],
            "outputInfo": [
                {"label": "训练清单", "value": "data/manifests/cifar10_train.csv"},
                {"label": "测试清单", "value": "data/manifests/cifar10_test.csv"},
                {"label": "图库清单", "value": "data/manifests/cifar10_gallery.csv"},
                {"label": "查询清单", "value": "data/manifests/cifar10_query.csv"},
                {"label": "摘要文件", "value": "data/manifests/cifar10_summary.json"},
                {"label": "最近更新", "value": summary_updated_at},
            ],
            "classes": class_rows,
            "manifestOptions": self._relative_paths(sorted((self.project_root / "data" / "manifests").glob("*.csv"))),
        }
        self._dataset_status_cache = (cache_key, payload)
        return payload

    def training_status(self) -> dict[str, Any]:
        history_path = self.metric_root / "training_history.json"
        payload = self._display_payload(self._read_json(history_path, {}))
        if isinstance(payload, list):
            history = self._normalize_history(payload)
            payload = {}
        else:
            history = self._normalize_history(payload.get("history", []))
        runtime = self._display_payload(self.training_service.runtime_status())
        runtime_history = self._normalize_history(runtime.get("history", []))
        if len(runtime_history) > len(history):
            history = runtime_history
        completed_history = [
            item for item in history if not str(item.get("status") or "").startswith(("训练中", "验证中"))
        ]
        latest = completed_history[-1] if completed_history else None
        display_history = self._training_history_with_runtime(history, runtime)
        checkpoints = self._relative_paths(self._preferred_checkpoint_paths())
        default_train_manifest = self.project_root / "data" / "manifests" / "cifar10_train.csv"
        train_manifests = [default_train_manifest] if default_train_manifest.exists() else sorted((self.project_root / "data" / "manifests").glob("*train*.csv"))
        if not train_manifests:
            train_manifests = sorted((self.project_root / "data" / "manifests").glob("*.csv"))
        default_validation_manifest = self.project_root / "data" / "manifests" / "cifar10_test.csv"
        validation_manifests = [default_validation_manifest] if default_validation_manifest.exists() else sorted((self.project_root / "data" / "manifests").glob("*test*.csv"))
        if not validation_manifests:
            validation_manifests = sorted((self.project_root / "data" / "manifests").glob("*.csv"))
        device_options = self.training_service.available_devices()
        optimizer_options = self.training_service.available_optimizers()
        selected_device = str(runtime.get("device") or payload.get("device") or self.training_service.device)
        if selected_device == "auto":
            selected_device = str(self.training_service.resolve_device("auto").type)
        class_names = payload.get("classNames") or self.CIFAR10_CLASSES
        latest_confusion_matrix = payload.get("latestConfusionMatrix", [])
        latest_per_class_metrics = payload.get("latestPerClassMetrics", [])
        best_confusion_matrix = payload.get("bestConfusionMatrix", latest_confusion_matrix)
        best_per_class_metrics = payload.get("bestPerClassMetrics", latest_per_class_metrics)
        best_metrics = payload.get("bestMetrics", {})
        return {
            "summary": {
                "latestEpoch": latest.get("epoch") if latest else None,
                "latestLoss": latest.get("loss") if latest else None,
                "latestAccuracy": latest.get("accuracy") if latest else None,
                "latestPrecision": latest.get("precision") if latest else None,
                "latestRecall": latest.get("recall") if latest else None,
                "latestMacroF1": latest.get("macroF1") if latest else None,
                "latestValLoss": latest.get("valLoss") if latest else None,
                "latestValAccuracy": latest.get("valAccuracy") if latest else None,
                "latestValPrecision": latest.get("valPrecision") if latest else None,
                "latestValRecall": latest.get("valRecall") if latest else None,
                "latestValMacroF1": latest.get("valMacroF1") if latest else None,
                "latestSampleCount": latest.get("sampleCount") if latest else None,
                "checkpointCount": len(checkpoints),
                "historyCount": len(completed_history),
                "device": selected_device,
                "isRunning": runtime.get("isRunning", False),
                "runtimeStatus": runtime.get("status"),
                "runtimeMessage": runtime.get("message"),
                "currentEpoch": runtime.get("currentEpoch"),
                "totalEpochs": runtime.get("totalEpochs"),
            },
            "modelInfo": [
                {"label": "骨干网络", "value": "ResNet101 Bottleneck 101层"},
                {"label": "特征图输出", "value": f"2048 x 7 x 7 -> {settings.feature_dim} x 7 x 7"},
                {"label": "注意力模块", "value": "CBAM通道注意力与空间得分图"},
                {"label": "自相似张量", "value": f"{settings.feature_dim} x 7 x 7 x 7 x 7"},
                {"label": "嵌入头", "value": "自相似张量编码 + 特征融合 + GeM"},
                {"label": "分类头", "value": "ArcFace scale 32 margin 0.2"},
                {"label": "微调策略", "value": "默认冻结ResNet101骨干，训练CBAM、自相似嵌入、ArcFace"},
                {"label": "训练设备", "value": selected_device},
                {"label": "默认权重", "value": "checkpoints/embedding_best.pt"},
            ],
            "modelArchitecture": [
                {"stage": "输入", "module": "RGB图像", "output": "3 x 224 x 224", "trainable": "--"},
                {"stage": "Stem", "module": "Conv7x7 + BN + ReLU + MaxPool", "output": "64 x 56 x 56", "trainable": "冻结"},
                {"stage": "ResNet layer1", "module": "Bottleneck x3", "output": "256 x 56 x 56", "trainable": "冻结"},
                {"stage": "ResNet layer2", "module": "Bottleneck x4", "output": "512 x 28 x 28", "trainable": "冻结"},
                {"stage": "ResNet layer3", "module": "Bottleneck x23", "output": "1024 x 14 x 14", "trainable": "冻结"},
                {"stage": "ResNet layer4", "module": "Bottleneck x3", "output": "2048 x 7 x 7", "trainable": "冻结"},
                {"stage": "通道映射", "module": "Conv1x1 + BN + ReLU", "output": f"{settings.feature_dim} x 7 x 7", "trainable": "训练"},
                {"stage": "CBAM", "module": "Channel Attention + Spatial Score", "output": f"{settings.feature_dim} x 7 x 7", "trainable": "训练"},
                {"stage": "自相似张量", "module": "Fs(x,p)=S(x) x F0(x+p)", "output": f"{settings.feature_dim} x 7 x 7 x 7 x 7", "trainable": "--"},
                {"stage": "张量编码", "module": "Conv3D(1x3x3) + BN + ReLU x3", "output": f"{settings.feature_dim} x 7 x 7", "trainable": "训练"},
                {"stage": "特征融合", "module": "BN(F0) + Fd -> Conv1x1 + ReLU + Conv1x1", "output": f"{settings.feature_dim} x 7 x 7", "trainable": "训练"},
                {"stage": "GeM池化", "module": "Generalized Mean Pooling 可学习p", "output": f"{settings.feature_dim}", "trainable": "训练"},
                {"stage": "分类头", "module": "ArcFace", "output": f"{len(class_names)}类logits", "trainable": "训练"},
            ],
            "trainingScheme": [
                {"label": "输入来源", "value": "按训练清单逐张读取 file_path"},
                {"label": "验证来源", "value": "按验证清单逐张读取 file_path"},
                {"label": "图像变换", "value": "Resize 224 x 224 / ToTensor / ImageNet Normalize"},
                {"label": "特征生成", "value": f"ResNet101 layer4 -> CBAM -> 自相似张量编码 -> 特征融合 -> GeM -> {settings.feature_dim}维嵌入"},
                {"label": "监督方式", "value": "ArcFace logits + CrossEntropyLoss"},
                {"label": "优化器", "value": payload.get("optimizer", "Adam")},
            ],
            "runInfo": [
                {"label": "训练清单", "value": payload.get("trainManifest") or runtime.get("trainManifest") or "--"},
                {"label": "验证清单", "value": payload.get("validationManifest") or runtime.get("validationManifest") or "--"},
                {"label": "训练设备", "value": selected_device or runtime.get("device") or "--"},
                {"label": "Epoch", "value": payload.get("epochs") or runtime.get("totalEpochs") or "--"},
                {"label": "Early Stop", "value": payload.get("earlyStopPatience") or "--"},
                {"label": "Batch Size", "value": payload.get("batchSize") or "--"},
                {"label": "Num Workers", "value": payload.get("numWorkers") if "numWorkers" in payload else "--"},
                {"label": "优化器", "value": payload.get("optimizer") or "--"},
                {"label": "学习率", "value": payload.get("learningRate") or "--"},
                {"label": "Seed", "value": payload.get("seed") if "seed" in payload else "--"},
                {
                    "label": "仅保存最佳",
                    "value": "--" if "saveBestOnly" not in payload else ("是" if payload.get("saveBestOnly") else "否"),
                },
                {
                    "label": "冻结骨干",
                    "value": "--" if "freezeBackbone" not in payload else ("是" if payload.get("freezeBackbone") else "否"),
                },
                {"label": "类别数", "value": len(class_names)},
            ],
            "evaluationSummary": {
                "latestValLoss": best_metrics.get("valLoss", latest.get("valLoss") if latest else None),
                "latestValAccuracy": best_metrics.get("valAccuracy", latest.get("valAccuracy") if latest else None),
                "latestValMacroF1": best_metrics.get("valMacroF1", latest.get("valMacroF1") if latest else None),
                "latestTrainAccuracy": latest.get("accuracy") if latest else None,
                "latestTrainMacroF1": latest.get("macroF1") if latest else None,
                "classCount": len(class_names),
                "device": selected_device,
                "bestEpoch": payload.get("bestEpoch"),
                "bestMonitorLoss": payload.get("bestMonitorLoss"),
                "isRunning": runtime.get("isRunning", False),
            },
            "latest": latest,
            "history": display_history,
            "classNames": class_names,
            "latestConfusionMatrix": best_confusion_matrix,
            "latestPerClassMetrics": best_per_class_metrics,
            "checkpointOptions": checkpoints,
            "deviceOptions": device_options,
            "optimizerOptions": optimizer_options,
            "manifestOptions": self._relative_paths(train_manifests),
            "validationManifestOptions": self._relative_paths(validation_manifests),
            "runtime": runtime,
        }

    def evaluation_status(self, selected_model: str | None = None) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []

        current_run = self._current_evaluation_run()
        if current_run:
            runs.append(current_run)

        runs.extend(self._archived_evaluation_runs())
        runs.extend(self._legacy_evaluation_runs())

        if not runs:
            return {
                "selectedModel": None,
                "modelOptions": [],
                "modelList": [],
                "summary": {
                    "latestValLoss": None,
                    "latestValAccuracy": None,
                    "latestValMacroF1": None,
                    "classCount": 0,
                    "bestEpoch": None,
                    "bestMonitorLoss": None,
                },
                "history": [],
                "classNames": [],
                "confusionMatrix": [],
                "perClassMetrics": [],
                "runInfo": [],
            }

        runs = sorted(runs, key=self._evaluation_run_sort_key)
        selected_run = next((item for item in runs if item["value"] == selected_model), runs[0])
        return {
            "selectedModel": selected_run["value"],
            "modelOptions": [{"label": item["optionLabel"], "value": item["value"]} for item in runs],
            "modelList": [self._evaluation_model_row(item) for item in runs],
            "summary": selected_run["summary"],
            "history": selected_run["history"],
            "classNames": selected_run["classNames"],
            "confusionMatrix": selected_run["confusionMatrix"],
            "perClassMetrics": selected_run["perClassMetrics"],
            "runInfo": selected_run["runInfo"],
        }

    def delete_model_weight(self, model_value: str) -> dict[str, Any]:
        value = str(model_value or "").strip()
        if not value:
            raise ValueError("模型不存在")
        if self.training_service.runtime_status().get("isRunning"):
            raise ValueError("训练任务进行中")

        if value == "current":
            deleted = 0
            for path in (
                self.project_root / "checkpoints" / "embedding_best.pt",
                self.project_root / "checkpoints" / "embedding_latest.pt",
                self.metric_root / "training_history.json",
            ):
                if path.exists():
                    path.unlink()
                    deleted += 1
            return {"deleted": deleted, "model": value}

        roots = {
            "archive:": self.training_service.training_runs_root,
            "legacy:": self.project_root / "visual-search-engine" / "data" / "processed" / "training_runs",
        }
        for prefix, root in roots.items():
            if value.startswith(prefix):
                run_name = value.removeprefix(prefix)
                target = (root / run_name).resolve()
                if not str(target).startswith(str(root.resolve())) or not target.exists() or not target.is_dir():
                    raise ValueError("模型不存在")
                shutil.rmtree(target)
                return {"deleted": 1, "model": value}

        checkpoint_path = resolve_project_path(value)
        checkpoint_root = (self.project_root / "checkpoints").resolve()
        if checkpoint_path.exists() and checkpoint_path.is_file() and str(checkpoint_path.resolve()).startswith(str(checkpoint_root)):
            checkpoint_path.unlink()
            return {"deleted": 1, "model": value}
        raise ValueError("模型不存在")

    def feature_status(self) -> dict[str, Any]:
        gallery_manifest = self.project_root / "features" / "cifar10_gallery_embedding.csv"
        query_manifest = self.project_root / "features" / "cifar10_query_embedding.csv"
        gallery_frame = self._read_csv_optional(gallery_manifest)
        query_frame = self._read_csv_optional(query_manifest)
        gallery_count = len(gallery_frame) if gallery_frame is not None else 0
        query_count = len(query_frame) if query_frame is not None else 0
        expected_gallery_count = self._csv_row_count(self.project_root / "data" / "manifests" / "cifar10_gallery.csv")
        expected_query_count = self._csv_row_count(self.project_root / "data" / "manifests" / "cifar10_query.csv")
        gallery_first = gallery_frame.iloc[0].to_dict() if gallery_frame is not None and not gallery_frame.empty else {}
        query_first = query_frame.iloc[0].to_dict() if query_frame is not None and not query_frame.empty else {}
        model_name = gallery_first.get("model_name") or query_first.get("model_name") or "--"
        gallery_architecture = str(gallery_first.get("architecture") or "")
        query_architecture = str(query_first.get("architecture") or "")
        architecture_ready = (
            gallery_architecture == SelfSimilarityEmbedding.architecture_name
            and query_architecture == SelfSimilarityEmbedding.architecture_name
        )
        updated_paths = [path for path in (gallery_manifest, query_manifest) if path.exists()]
        updated_at = self._format_timestamp(max(updated_paths, key=lambda path: path.stat().st_mtime)) if updated_paths else "--"
        if gallery_count == 0 and query_count == 0:
            status = "未生成"
        elif gallery_count == expected_gallery_count and query_count == expected_query_count and architecture_ready:
            status = "完整"
        else:
            status = "不完整"
        records = [
            {
                "name": "自相似嵌入特征集",
                "filePath": "features/cifar10_gallery_embedding.csv|features/cifar10_query_embedding.csv",
                "mode": "embedding",
                "modeLabel": "自相似嵌入特征",
                "dimension": settings.feature_dim,
                "modelName": model_name,
                "galleryCount": gallery_count,
                "queryCount": query_count,
                "expectedGalleryCount": expected_gallery_count,
                "expectedQueryCount": expected_query_count,
                "status": status,
                "updatedAt": updated_at,
            }
        ]
        checkpoint_options = self._relative_paths(self._preferred_checkpoint_paths())
        return {
            "summary": {
                "recordCount": 1 if status != "未生成" else 0,
                "latestName": "自相似嵌入特征集" if status != "未生成" else None,
                "latestCount": gallery_count + query_count if status != "未生成" else None,
                "galleryCount": gallery_count,
                "queryCount": query_count,
                "checkpointCount": len(checkpoint_options),
                "currentModel": self._current_model_name(),
            },
            "modeInfo": [],
            "pipelineInfo": [],
            "records": records,
            "manifestOptions": [],
            "featureManifestOptions": self._relative_paths(sorted((self.project_root / "features").glob("*.csv"))),
            "checkpointOptions": checkpoint_options,
        }

    def experiment_status(self) -> dict[str, Any]:
        records = []
        complete_run_ids = {group["runId"] for group in self._complete_matrix_record_groups()}
        for result_path in sorted(self.metric_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True):
            if result_path.name in {"training_history.json", "training_runtime.json", "duplicate_threshold_scan.json", "offline_experiment_runtime.json"}:
                continue
            payload = self._read_json(result_path, {})
            index_type = str(payload.get("indexType") or "")
            if result_path.name.startswith("matrix_") and index_type not in self.MATRIX_INDEX_METHODS:
                continue
            if (
                result_path.name.startswith("matrix_")
                and str(payload.get("featureScheme") or "") == "embedding"
                and str(payload.get("featureArchitecture") or "") != SelfSimilarityEmbedding.architecture_name
            ):
                continue
            feature_label = payload.get("featureLabel") or self._experiment_feature_label(result_path.name)
            run_id = payload.get("runId") or f"{self._compact_timestamp(result_path)}_{payload.get('featureScheme') or self._experiment_feature_scheme(result_path.name)}_{'rerank' if payload.get('rerank') else 'plain'}"
            if result_path.name.startswith("matrix_") and run_id not in complete_run_ids:
                continue
            run_label = payload.get("runLabel") or f"{self._compact_timestamp(result_path)} {self._feature_display_name(feature_label)} {'重排序' if payload.get('rerank') else '未重排序'}"
            records.append(
                {
                    "name": result_path.name,
                    "displayName": self._experiment_display_name(result_path.name, payload.get("indexType")),
                    "runId": run_id,
                    "runLabel": run_label,
                    "createdAt": payload.get("createdAt") or self._format_timestamp(result_path),
                    "featureScheme": payload.get("featureScheme"),
                    "featureLabel": self._feature_display_name(feature_label),
                    "featureModelName": payload.get("featureModelName"),
                    "featureArchitecture": payload.get("featureArchitecture"),
                    "indexLabel": self._index_display_name(payload.get("indexType")),
                    "indexType": payload.get("indexType", "--"),
                    "indexMethod": payload.get("indexMethod", "--"),
                    "indexLibrary": payload.get("indexLibrary", "--"),
                    "topK": payload.get("topK", "--"),
                    "galleryCount": payload.get("galleryCount", "--"),
                    "queryCount": payload.get("queryCount", "--"),
                    "mapAtK": payload.get("mapAtK", "--"),
                    "recallAtK": payload.get("recallAtK", "--"),
                    "precisionAtK": payload.get("precisionAtK", "--"),
                    "elapsedMs": payload.get("averageElapsedMs", "--"),
                    "indexSizeBytes": payload.get("indexSizeBytes", "--"),
                    "rerank": "是" if payload.get("rerank") else "否",
                    "updatedAt": self._format_timestamp(result_path),
                }
            )
        feature_options = self._relative_paths(sorted((self.project_root / "features").glob("*.csv")))
        return {
            "records": records,
            "featureManifestOptions": feature_options,
        }

    def _complete_matrix_record_groups(
        self,
        feature_scheme: str | None = None,
        rerank_enabled: bool | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for result_path in self.metric_root.glob("matrix_*.json"):
            payload = self._read_json(result_path, {})
            index_type = str(payload.get("indexType") or "")
            run_id = str(payload.get("runId") or "")
            if index_type not in self.MATRIX_INDEX_METHODS or not run_id:
                continue
            if feature_scheme is not None and str(payload.get("featureScheme") or "") != feature_scheme:
                continue
            if (
                str(payload.get("featureScheme") or "") == "embedding"
                and str(payload.get("featureArchitecture") or "") != SelfSimilarityEmbedding.architecture_name
            ):
                continue
            if rerank_enabled is not None and bool(payload.get("rerank")) != bool(rerank_enabled):
                continue
            if top_k is not None and int(payload.get("topK") or 0) != int(top_k):
                continue
            if not self._is_valid_metric_record(payload):
                continue
            group = grouped.setdefault(
                run_id,
                {
                    "runId": run_id,
                    "featureScheme": payload.get("featureScheme"),
                    "rerank": bool(payload.get("rerank")),
                    "topK": payload.get("topK"),
                    "updatedAt": result_path.stat().st_mtime,
                    "indexes": set(),
                },
            )
            group["indexes"].add(index_type)
            group["updatedAt"] = max(float(group["updatedAt"]), result_path.stat().st_mtime)
        complete = [
            group
            for group in grouped.values()
            if set(self.MATRIX_INDEX_METHODS).issubset(group["indexes"])
        ]
        return sorted(complete, key=lambda item: float(item["updatedAt"]), reverse=True)

    @staticmethod
    def _is_valid_metric_record(payload: dict[str, Any]) -> bool:
        try:
            float(payload.get("mapAtK"))
            float(payload.get("recallAtK"))
            float(payload.get("averageElapsedMs"))
            int(payload.get("indexSizeBytes"))
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _experiment_display_name(filename: str, index_type: str | None = None) -> str:
        index_label = OfflinePipelineService._index_display_name(index_type)
        if filename == "exp1_baseline.json":
            return f"ResNet101+{index_label or '基础检索'}"
        if filename == "exp2_baseline.json":
            return f"ResNet101+{index_label or '基础检索'}"
        if filename == "exp2_embedding.json":
            return f"自相似特征+{index_label or '基础检索'}"
        if filename == "exp3_hnsw.json":
            return f"自相似特征+{index_label or 'HNSW'}"
        if filename == "exp3_hnsw_rerank.json":
            return f"自相似特征+{index_label or 'HNSW'}+重排序"
        if filename.startswith("exp4_"):
            return f"自相似特征+{index_label or filename.removeprefix('exp4_').removesuffix('.json')}"
        return filename

    @staticmethod
    def _index_display_name(index_type: str | None) -> str:
        return {
            "brute": "暴力索引",
            "kd_tree": "KD-Tree",
            "hnsw": "HNSW",
            "pq": "PQ",
        }.get(str(index_type or ""), str(index_type or "--"))

    @staticmethod
    def _experiment_feature_label(filename: str) -> str:
        if "embedding" in filename:
            return "自相似特征"
        if "baseline" in filename:
            return "ResNet101"
        return "--"

    @staticmethod
    def _experiment_feature_scheme(filename: str) -> str:
        if "embedding" in filename:
            return "embedding"
        if "baseline" in filename:
            return "baseline"
        return "unknown"

    @staticmethod
    def _feature_display_name(value: str) -> str:
        return {
            "ResNet101特征嵌入": "ResNet101",
            "模型特征嵌入": "自相似特征",
            "自相似嵌入特征": "自相似特征",
        }.get(str(value or ""), str(value or "--"))

    def _resolve_required(self, path_value: str | None) -> Path:
        if not path_value:
            raise ValueError("缺少清单路径")
        return self._resolve_path(path_value)

    def _resolve_path(self, path_value: str) -> Path:
        return resolve_project_path(path_value)

    def _feature_manifest_source_options(self) -> list[str]:
        manifest_root = self.project_root / "data" / "manifests"
        preferred = [
            manifest_root / "cifar10_gallery.csv",
            manifest_root / "cifar10_query.csv",
        ]
        return self._relative_paths([path for path in preferred if path.exists()])

    @staticmethod
    def _manifest_source_label(partition: str, manifest_name: str) -> str:
        text = f"{partition} {manifest_name}".lower()
        if "gallery" in text:
            return "图库图片"
        if "query" in text:
            return "查询图片"
        if "train" in text:
            return "训练图片"
        if "test" in text:
            return "测试图片"
        return partition or "--"

    def _preferred_checkpoint_paths(self) -> list[Path]:
        checkpoint_root = self.project_root / "checkpoints"
        preferred_names = ("embedding_best.pt", "embedding_latest.pt")
        paths = [checkpoint_root / name for name in preferred_names if (checkpoint_root / name).exists()]
        preferred_set = {path.resolve() for path in paths}
        others = [
            path
            for path in sorted(checkpoint_root.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
            if path.resolve() not in preferred_set
        ]
        return paths + others

    def _current_model_name(self) -> str:
        payload = self._read_json(self.metric_root / "training_history.json", {})
        runtime = self.training_service.runtime_status()
        run_id = None
        if isinstance(payload, dict):
            run_id = payload.get("runId")
        run_id = self._first_present(run_id, runtime.get("runId"))
        return self._display_run_id(run_id) if run_id else "--"

    def _current_evaluation_run(self) -> dict[str, Any] | None:
        history_path = self.metric_root / "training_history.json"
        raw_payload = self._read_json(history_path, {})
        payload = self._display_payload(raw_payload)
        runtime = self._display_payload(self.training_service.runtime_status())

        if isinstance(payload, list):
            history = self._normalize_history(payload)
            payload = {}
        else:
            history = self._normalize_history(payload.get("history", []))
        runtime_history = self._normalize_history(runtime.get("history", []))
        if len(runtime_history) > len(history):
            history = runtime_history

        has_payload = bool(payload) or bool(history) or bool(runtime.get("runId"))
        if not has_payload:
            return None

        checkpoint_path = None
        for checkpoint_name in ("embedding_best.pt", "embedding_latest.pt"):
            candidate = self.project_root / "checkpoints" / checkpoint_name
            if candidate.exists():
                checkpoint_path = self._relative_path(candidate)
                break

        best_metrics = self._normalize_history_record(payload.get("bestMetrics", {}))
        best_epoch = self._first_present(payload.get("bestEpoch"), best_metrics.get("epoch"))
        return self._build_evaluation_run(
            value="current",
            source_label="当前模型",
            run_id=self._first_present(payload.get("runId"), runtime.get("runId"), "current"),
            checkpoint_path=checkpoint_path,
            history=history,
            class_names=payload.get("classNames") or self.CIFAR10_CLASSES,
            confusion_matrix=payload.get("bestConfusionMatrix") or payload.get("latestConfusionMatrix") or [],
            per_class_metrics=payload.get("bestPerClassMetrics") or payload.get("latestPerClassMetrics") or [],
            best_metrics=best_metrics,
            metadata={
                "sourcePath": self._relative_path(history_path),
                "trainManifest": self._first_present(payload.get("trainManifest"), runtime.get("trainManifest")),
                "validationManifest": self._first_present(payload.get("validationManifest"), runtime.get("validationManifest")),
                "epochs": self._first_present(payload.get("epochs"), runtime.get("totalEpochs")),
                "completedEpochs": len(history),
                "earlyStopPatience": payload.get("earlyStopPatience"),
                "batchSize": payload.get("batchSize"),
                "numWorkers": payload.get("numWorkers"),
                "optimizer": payload.get("optimizer"),
                "learningRate": payload.get("learningRate"),
                "seed": payload.get("seed"),
                "saveBestOnly": payload.get("saveBestOnly"),
                "freezeBackbone": payload.get("freezeBackbone"),
                "bestEpoch": best_epoch,
                "bestMonitorLoss": payload.get("bestMonitorLoss"),
                "startedAt": runtime.get("startedAt"),
                "finishedAt": runtime.get("finishedAt"),
                "status": runtime.get("status"),
            },
        )

    def _archived_evaluation_runs(self) -> list[dict[str, Any]]:
        run_root = self.training_service.training_runs_root
        if not run_root.exists():
            return []

        runs: list[dict[str, Any]] = []
        for run_dir in sorted((path for path in run_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime_ns, reverse=True):
            metrics_payload = self._display_payload(self._read_json(run_dir / "metrics.json", {}))
            summary_payload = self._display_payload(self._read_json(run_dir / "summary.json", {}))
            runtime_payload = self._display_payload(self._read_json(run_dir / "runtime.json", {}))
            history = self._normalize_history(metrics_payload.get("history", []))

            if not metrics_payload and not summary_payload and not runtime_payload and not history:
                continue

            checkpoint_path = None
            for checkpoint_name in ("embedding_best.pt", "embedding_latest.pt"):
                candidate = run_dir / checkpoint_name
                if candidate.exists():
                    checkpoint_path = self._relative_path(candidate)
                    break

            best_metrics = self._normalize_history_record(metrics_payload.get("bestMetrics", {}))
            best_epoch = self._first_present(metrics_payload.get("bestEpoch"), summary_payload.get("bestEpoch"), best_metrics.get("epoch"))
            runs.append(
                self._build_evaluation_run(
                    value=f"archive:{run_dir.name}",
                    source_label="归档模型",
                    run_id=self._first_present(metrics_payload.get("runId"), summary_payload.get("runId"), run_dir.name),
                    checkpoint_path=checkpoint_path,
                    history=history,
                    class_names=metrics_payload.get("classNames") or self.CIFAR10_CLASSES,
                    confusion_matrix=metrics_payload.get("bestConfusionMatrix") or metrics_payload.get("latestConfusionMatrix") or [],
                    per_class_metrics=metrics_payload.get("bestPerClassMetrics") or metrics_payload.get("latestPerClassMetrics") or [],
                    best_metrics=best_metrics,
                    metadata={
                        "sourcePath": self._relative_path(run_dir),
                        "trainManifest": self._first_present(metrics_payload.get("trainManifest"), summary_payload.get("trainManifest"), runtime_payload.get("trainManifest")),
                        "validationManifest": self._first_present(metrics_payload.get("validationManifest"), summary_payload.get("validationManifest"), runtime_payload.get("validationManifest")),
                        "epochs": self._first_present(metrics_payload.get("epochs"), runtime_payload.get("totalEpochs")),
                        "completedEpochs": len(history),
                        "earlyStopPatience": metrics_payload.get("earlyStopPatience"),
                        "batchSize": metrics_payload.get("batchSize"),
                        "numWorkers": metrics_payload.get("numWorkers"),
                        "optimizer": metrics_payload.get("optimizer"),
                        "learningRate": metrics_payload.get("learningRate"),
                        "seed": metrics_payload.get("seed"),
                        "saveBestOnly": metrics_payload.get("saveBestOnly"),
                        "freezeBackbone": metrics_payload.get("freezeBackbone"),
                        "bestEpoch": best_epoch,
                        "bestMonitorLoss": metrics_payload.get("bestMonitorLoss"),
                        "startedAt": self._first_present(summary_payload.get("startedAt"), runtime_payload.get("startedAt")),
                        "finishedAt": self._first_present(summary_payload.get("finishedAt"), runtime_payload.get("finishedAt")),
                        "status": self._first_present(summary_payload.get("status"), runtime_payload.get("status")),
                    },
                )
            )
        return runs

    def _legacy_evaluation_runs(self) -> list[dict[str, Any]]:
        legacy_root = self.project_root / "visual-search-engine" / "data" / "processed" / "training_runs"
        if not legacy_root.exists():
            return []

        runs: list[dict[str, Any]] = []
        for run_dir in sorted((path for path in legacy_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime_ns, reverse=True):
            summary_payload = self._display_payload(self._read_json(run_dir / "summary.json", {}))
            history_payload = self._read_json(run_dir / "history.json", [])
            history = self._normalize_history(history_payload if isinstance(history_payload, list) else [])

            if not summary_payload and not history:
                continue

            runs.append(
                self._build_evaluation_run(
                    value=f"legacy:{run_dir.name}",
                    source_label="历史训练",
                    run_id=self._first_present(summary_payload.get("run_id"), run_dir.name),
                    checkpoint_path=summary_payload.get("checkpoint_path"),
                    history=history,
                    class_names=self.CIFAR10_CLASSES,
                    confusion_matrix=[],
                    per_class_metrics=[],
                    best_metrics={},
                    metadata={
                        "sourcePath": self._relative_path(run_dir),
                        "epochs": self._first_present(summary_payload.get("epochs"), summary_payload.get("completed_epochs")),
                        "completedEpochs": self._first_present(summary_payload.get("completed_epochs"), len(history)),
                        "batchSize": summary_payload.get("batch_size"),
                        "numWorkers": summary_payload.get("num_workers"),
                        "learningRate": summary_payload.get("learning_rate"),
                        "seed": summary_payload.get("seed"),
                        "bestEpoch": self._best_history_record(history).get("epoch") if history else None,
                        "startedAt": summary_payload.get("started_at"),
                        "finishedAt": summary_payload.get("finished_at"),
                        "status": summary_payload.get("status"),
                    },
                )
            )
        return runs

    def _build_evaluation_run(
        self,
        *,
        value: str,
        source_label: str,
        run_id: str,
        checkpoint_path: str | None,
        history: list[dict[str, Any]],
        class_names: list[str],
        confusion_matrix: list[list[int]],
        per_class_metrics: list[dict[str, Any]],
        best_metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        latest = history[-1] if history else {}
        best_record = self._best_history_record(history)
        best_val_loss = self._first_present(best_metrics.get("valLoss"), best_record.get("valLoss"), latest.get("valLoss"))
        best_val_accuracy = self._first_present(best_metrics.get("valAccuracy"), best_record.get("valAccuracy"), latest.get("valAccuracy"))
        best_val_macro_f1 = self._first_present(best_metrics.get("valMacroF1"), best_record.get("valMacroF1"), latest.get("valMacroF1"))
        option_label = self._evaluation_option_label(source_label, run_id, best_val_accuracy)

        run_info = [{"label": "模型来源", "value": source_label}]
        self._append_info(run_info, "模型名称", self._display_run_id(run_id))
        self._append_info(run_info, "权重来源", self._checkpoint_source_label(checkpoint_path))
        self._append_info(run_info, "来源目录", metadata.get("sourcePath"))
        self._append_info(run_info, "训练清单", metadata.get("trainManifest"))
        self._append_info(run_info, "验证清单", metadata.get("validationManifest"))
        self._append_info(run_info, "最佳 Epoch", self._first_present(metadata.get("bestEpoch"), best_record.get("epoch")))
        self._append_info(run_info, "Epoch", self._first_present(metadata.get("epochs"), metadata.get("completedEpochs"), len(history) if history else None))
        self._append_info(run_info, "Early Stop", metadata.get("earlyStopPatience"))
        self._append_info(run_info, "Batch Size", metadata.get("batchSize"))
        self._append_info(run_info, "Num Workers", metadata.get("numWorkers"))
        self._append_info(run_info, "优化器", metadata.get("optimizer"))
        self._append_info(run_info, "学习率", metadata.get("learningRate"))
        self._append_info(run_info, "Seed", metadata.get("seed"))
        if metadata.get("saveBestOnly") is not None:
            self._append_info(run_info, "仅保存最佳", "是" if metadata.get("saveBestOnly") else "否")
        if metadata.get("freezeBackbone") is not None:
            self._append_info(run_info, "冻结骨干", "是" if metadata.get("freezeBackbone") else "否")
        self._append_info(run_info, "完成时间", metadata.get("finishedAt"))
        self._append_info(run_info, "状态", metadata.get("status"))
        self._append_info(run_info, "类别数", len(class_names))

        return {
            "value": value,
            "optionLabel": option_label,
            "modelName": option_label,
            "sourceLabel": source_label,
            "runId": run_id,
            "checkpointPath": checkpoint_path,
            "deletable": True,
            "summary": {
                "latestValLoss": best_val_loss,
                "latestValAccuracy": best_val_accuracy,
                "latestValMacroF1": best_val_macro_f1,
                "classCount": len(class_names),
                "bestEpoch": self._first_present(metadata.get("bestEpoch"), best_record.get("epoch")),
                "bestMonitorLoss": metadata.get("bestMonitorLoss"),
            },
            "history": history,
            "classNames": class_names,
            "confusionMatrix": confusion_matrix,
            "perClassMetrics": per_class_metrics,
            "runInfo": run_info,
        }

    @staticmethod
    def _evaluation_run_sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
        accuracy = item.get("summary", {}).get("latestValAccuracy")
        try:
            accuracy_value = float(accuracy)
            has_accuracy = 1
        except (TypeError, ValueError):
            accuracy_value = -1.0
            has_accuracy = 0
        return (-has_accuracy, -accuracy_value, str(item.get("modelName") or item.get("optionLabel") or ""))

    @staticmethod
    def _evaluation_model_row(item: dict[str, Any]) -> dict[str, Any]:
        summary = item.get("summary", {})
        run_info = item.get("runInfo", [])
        status = next((entry.get("value") for entry in run_info if entry.get("label") == "状态"), "--")
        return {
            "value": item.get("value"),
            "modelName": item.get("modelName") or item.get("optionLabel"),
            "sourceLabel": item.get("sourceLabel"),
            "runId": item.get("runId"),
            "checkpointPath": item.get("checkpointPath"),
            "deletable": bool(item.get("deletable", True)),
            "valLoss": summary.get("latestValLoss"),
            "valAccuracy": summary.get("latestValAccuracy"),
            "valMacroF1": summary.get("latestValMacroF1"),
            "bestEpoch": summary.get("bestEpoch"),
            "status": status,
        }

    def _relative_paths(self, paths: list[Path]) -> list[str]:
        return [self._relative_path(path) for path in paths]

    def _relative_path(self, path: Path) -> str:
        return serialize_project_path(path)

    def _display_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._display_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._display_payload(item) for item in value]
        if isinstance(value, str):
            return self._display_path_value(value)
        return value

    def _display_path_value(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            return value.replace("\\", "/")
        try:
            return self._relative_path(path)
        except ValueError:
            return value.replace("\\", "/")

    def _dataset_cache_key(self) -> tuple[tuple[str, int], ...]:
        manifest_root = self.project_root / "data" / "manifests"
        keys = []
        for path in sorted(manifest_root.glob("*.csv")):
            keys.append((path.name, path.stat().st_mtime_ns))
        summary_path = manifest_root / "cifar10_summary.json"
        if summary_path.exists():
            keys.append((summary_path.name, summary_path.stat().st_mtime_ns))
        return tuple(keys)

    def _raw_cifar10_metadata(self) -> dict[str, Any]:
        raw_root = self.project_root / "data" / "raw" / "cifar10" / "cifar-10-batches-py"
        downloaded = raw_root.exists()
        return {
            "trainCount": 50000,
            "testCount": 10000,
            "classes": self.CIFAR10_CLASSES,
            "downloaded": downloaded,
        }

    @staticmethod
    def _read_csv_optional(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        return read_csv(path)

    @staticmethod
    def _csv_row_count(path: Path) -> int:
        frame = OfflinePipelineService._read_csv_optional(path)
        return len(frame) if frame is not None else 0

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _normalize_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [OfflinePipelineService._normalize_history_record(item) for item in history if isinstance(item, dict)]

    @staticmethod
    def _normalize_history_record(item: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        status = item.get("status")
        if not status and item.get("epoch") is not None:
            status = "完成"
        return {
            "epoch": item.get("epoch"),
            "status": status,
            "loss": OfflinePipelineService._first_present(item.get("loss"), item.get("train_loss")),
            "accuracy": OfflinePipelineService._first_present(item.get("accuracy"), item.get("train_accuracy")),
            "precision": OfflinePipelineService._first_present(item.get("precision"), item.get("train_precision")),
            "recall": OfflinePipelineService._first_present(item.get("recall"), item.get("train_recall")),
            "macroF1": OfflinePipelineService._first_present(item.get("macroF1"), item.get("train_macro_f1")),
            "valLoss": OfflinePipelineService._first_present(item.get("valLoss"), item.get("val_loss")),
            "valAccuracy": OfflinePipelineService._first_present(item.get("valAccuracy"), item.get("val_accuracy")),
            "valPrecision": OfflinePipelineService._first_present(item.get("valPrecision"), item.get("val_precision")),
            "valRecall": OfflinePipelineService._first_present(item.get("valRecall"), item.get("val_recall")),
            "valMacroF1": OfflinePipelineService._first_present(item.get("valMacroF1"), item.get("val_macro_f1")),
            "sampleCount": item.get("sampleCount"),
            "totalSampleCount": item.get("totalSampleCount"),
            "learningRate": OfflinePipelineService._first_present(item.get("learningRate"), item.get("learning_rate")),
        }

    def _training_history_with_runtime(self, history: list[dict[str, Any]], runtime: dict[str, Any]) -> list[dict[str, Any]]:
        status = str(runtime.get("status") or "")
        if status == "idle":
            return history
        current_epoch = int(runtime.get("currentEpoch") or 0)
        completed_epochs = [
            int(item.get("epoch"))
            for item in history
            if str(item.get("epoch") or "").isdigit() and item.get("status") == "完成"
        ]
        latest_completed_epoch = max(completed_epochs) if completed_epochs else 0
        has_runtime_row = any(str(item.get("status") or "").startswith(("训练中", "验证中")) for item in history)
        if has_runtime_row:
            return history
        if runtime.get("isRunning") and current_epoch > latest_completed_epoch:
            return [
                *history,
                {
                    "epoch": current_epoch or "准备",
                    "status": runtime.get("message") or "训练中",
                    "loss": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "macroF1": None,
                    "valLoss": None,
                    "valAccuracy": None,
                    "valPrecision": None,
                    "valRecall": None,
                    "valMacroF1": None,
                    "sampleCount": "--",
                    "learningRate": None,
                },
            ]
        if not history and status in {"failed", "stopped", "interrupted"}:
            return [
                {
                    "epoch": current_epoch or "--",
                    "status": runtime.get("message") or status,
                    "loss": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "macroF1": None,
                    "valLoss": None,
                    "valAccuracy": None,
                    "valPrecision": None,
                    "valRecall": None,
                    "valMacroF1": None,
                    "sampleCount": "--",
                    "learningRate": None,
                }
            ]
        return history

    @staticmethod
    def _count_label(dataframe: pd.DataFrame | None, label_name: str) -> int:
        if dataframe is None or dataframe.empty:
            return 0
        return int((dataframe["label_name"] == label_name).sum())

    @staticmethod
    def _best_history_record(history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {}
        candidates = [item for item in history if item.get("valAccuracy") is not None]
        if candidates:
            return max(candidates, key=lambda item: float(item.get("valAccuracy", 0)))
        return history[-1]

    @staticmethod
    def _evaluation_option_label(source_label: str, run_id: str, best_val_accuracy: Any) -> str:
        accuracy_text = "--"
        if best_val_accuracy is not None:
            try:
                accuracy_text = f"{float(best_val_accuracy) * 100:.2f}%"
            except (TypeError, ValueError):
                accuracy_text = str(best_val_accuracy)
        model_name = OfflinePipelineService._display_run_id(run_id)
        if model_name not in {"--", "current"}:
            return f"{model_name} | {accuracy_text}"
        if source_label == "当前模型":
            return f"{source_label} | {accuracy_text}"
        return f"{model_name} | {accuracy_text}"

    @staticmethod
    def _display_run_id(run_id: Any) -> str:
        if run_id is None or run_id == "":
            return "--"
        text = str(run_id)
        return text.replace("_", "")

    @staticmethod
    def _checkpoint_source_label(checkpoint_path: Any) -> str:
        if not checkpoint_path:
            return "--"
        name = Path(str(checkpoint_path)).name
        if name == "embedding_best.pt":
            return "最佳权重"
        if name == "embedding_latest.pt":
            return "最新权重"
        return name

    @staticmethod
    def _append_info(items: list[dict[str, Any]], label: str, value: Any) -> None:
        if value is None or value == "":
            return
        items.append({"label": label, "value": value})

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if value == "":
                continue
            return value
        return None

    @staticmethod
    def _format_timestamp(path: Path) -> str:
        return path.stat().st_mtime_ns and str(pd.Timestamp(path.stat().st_mtime_ns, unit="ns").tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"))

    def _format_optional_timestamp(self, path: Path) -> str:
        if not path.exists():
            return "--"
        return self._format_timestamp(path)

    @staticmethod
    def _compact_timestamp(path: Path) -> str:
        return str(pd.Timestamp(path.stat().st_mtime_ns, unit="ns").tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%Y%m%d%H%M%S"))
