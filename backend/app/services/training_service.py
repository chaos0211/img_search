from __future__ import annotations

import json
import random
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.app.config import settings
from backend.app.models.arcface_head import ArcFaceHead
from backend.app.models.resnet_backbone import ResNet101Backbone
from backend.app.models.self_similarity_embedding import SelfSimilarityEmbedding
from backend.app.utils.file_utils import read_csv, resolve_project_path, serialize_project_path, write_csv


class TrainingStopped(Exception):
    pass


class ManifestImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transforms):
        self.records = records
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(resolve_project_path(record["file_path"])).convert("RGB")
        tensor = self.transforms(image)
        return tensor, int(record["label_index"])


class EmbeddingTrainingService:
    def __init__(self):
        self.device = self.resolve_device("auto")
        self.checkpoint_root = settings.project_root / "checkpoints"
        self.metric_root = settings.output_root / "metrics"
        self.training_runs_root = settings.output_root / "training_runs"
        self.feature_manifest_root = settings.project_root / "features"
        self.runtime_path = self.metric_root / "training_runtime.json"
        self.history_path = self.metric_root / "training_history.json"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.metric_root.mkdir(parents=True, exist_ok=True)
        self.training_runs_root.mkdir(parents=True, exist_ok=True)
        self.feature_manifest_root.mkdir(parents=True, exist_ok=True)
        self._task_lock = threading.Lock()
        self._task_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @staticmethod
    def _mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    @staticmethod
    def _cuda_available() -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _cuda_device_count() -> int:
        if not EmbeddingTrainingService._cuda_available():
            return 0
        try:
            return int(torch.cuda.device_count())
        except Exception:
            return 0

    @staticmethod
    def _cuda_device_label(index: int = 0) -> str:
        try:
            name = str(torch.cuda.get_device_name(index)).strip()
        except Exception:
            name = ""
        return f"CUDA {index} {name}".strip() if name else f"CUDA {index}"

    def available_devices(self) -> list[dict[str, Any]]:
        cuda_available = self._cuda_available()
        cuda_count = self._cuda_device_count()
        devices: list[dict[str, Any]] = [{"label": "自动", "value": "auto", "available": True}]
        devices.append(
            {
                "label": self._cuda_device_label(0) if cuda_available and cuda_count == 1 else "CUDA",
                "value": "cuda",
                "available": cuda_available,
                "deviceCount": cuda_count,
            }
        )
        if cuda_available and cuda_count > 1:
            for index in range(cuda_count):
                devices.append(
                    {
                        "label": self._cuda_device_label(index),
                        "value": f"cuda:{index}",
                        "available": True,
                        "deviceCount": cuda_count,
                    }
                )
        if self._mps_available():
            devices.append({"label": "MPS", "value": "mps", "available": True})
        devices.append({"label": "CPU", "value": "cpu", "available": True})
        return devices

    @staticmethod
    def available_optimizers() -> list[dict[str, str]]:
        return [
            {"label": "Adam", "value": "adam"},
            {"label": "AdamW", "value": "adamw"},
            {"label": "SGD", "value": "sgd"},
        ]

    def resolve_device(self, device_name: str | None) -> torch.device:
        target = (device_name or "auto").strip().lower()
        if target == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if self._mps_available():
                return torch.device("mps")
            return torch.device("cpu")
        if target == "cpu":
            return torch.device("cpu")
        if target == "mps":
            if not self._mps_available():
                raise ValueError("MPS 当前不可用")
            return torch.device("mps")
        if target == "cuda" or target.startswith("cuda:"):
            if not self._cuda_available():
                raise ValueError("CUDA 当前不可用")
            if ":" in target:
                try:
                    index = int(target.split(":", 1)[1])
                except ValueError as exc:
                    raise ValueError("CUDA 设备不存在") from exc
                if index < 0 or index >= self._cuda_device_count():
                    raise ValueError("CUDA 设备不存在")
            return torch.device(target)
        raise ValueError("训练设备不存在")

    def runtime_status(self) -> dict[str, Any]:
        default = {
            "isRunning": False,
            "status": "idle",
            "message": None,
            "currentEpoch": 0,
            "totalEpochs": 0,
            "runId": None,
            "trainManifest": None,
            "validationManifest": None,
            "device": None,
            "history": [],
            "startedAt": None,
            "updatedAt": None,
            "finishedAt": None,
        }
        if not self.runtime_path.exists():
            return default
        try:
            payload = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        except Exception:
            return default
        if not isinstance(payload, dict):
            return default
        default.update(payload)
        default = self._serialize_for_output(default)
        if default.get("isRunning") and not (self._task_thread and self._task_thread.is_alive()):
            default.update(
                {
                    "isRunning": False,
                    "status": "interrupted",
                    "message": "训练任务已中断，状态已重置",
                    "updatedAt": self._now(),
                    "finishedAt": default.get("finishedAt") or self._now(),
                }
            )
            self._write_runtime(default)
        return default

    def start_training(
        self,
        train_manifest_path: str,
        validation_manifest_path: str | None = None,
        epochs: int = 10,
        early_stop_patience: int = 3,
        batch_size: int = 16,
        num_workers: int = 2,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42,
        save_best_only: bool = True,
        freeze_backbone: bool = True,
        device_name: str = "auto",
    ) -> dict[str, Any]:
        with self._task_lock:
            if self.runtime_status().get("isRunning"):
                raise ValueError("训练任务进行中")
            self._stop_event.clear()
            started_at = self._now()
            run_id = self._run_id()
            self._write_metrics_payload(
                {
                    "runId": run_id,
                    "device": device_name,
                    "trainManifest": train_manifest_path,
                    "validationManifest": validation_manifest_path,
                    "epochs": epochs,
                    "earlyStopPatience": early_stop_patience,
                    "batchSize": batch_size,
                    "numWorkers": num_workers,
                    "optimizer": optimizer_name,
                    "learningRate": learning_rate,
                    "seed": seed,
                    "saveBestOnly": save_best_only,
                    "freezeBackbone": freeze_backbone,
                    "architecture": SelfSimilarityEmbedding.architecture_name,
                    "embeddingConfig": SelfSimilarityEmbedding(target_dim=settings.feature_dim).config(),
                    "classNames": [],
                    "history": [],
                    "bestEpoch": None,
                    "bestMonitorLoss": None,
                    "bestMetrics": {},
                    "bestConfusionMatrix": [],
                    "bestPerClassMetrics": [],
                    "latestConfusionMatrix": [],
                    "latestPerClassMetrics": [],
                    "stoppedEarly": False,
                }
            )
            self._write_runtime(
                {
                    "isRunning": True,
                    "status": "queued",
                    "message": "训练已启动",
                    "currentEpoch": 0,
                    "totalEpochs": epochs,
                    "runId": run_id,
                    "trainManifest": train_manifest_path,
                    "validationManifest": validation_manifest_path,
                    "device": device_name,
                    "history": [],
                    "startedAt": started_at,
                    "updatedAt": started_at,
                    "finishedAt": None,
                }
            )
            self._task_thread = threading.Thread(
                target=self._run_training_task,
                kwargs={
                    "train_manifest_path": train_manifest_path,
                    "validation_manifest_path": validation_manifest_path,
                    "epochs": epochs,
                    "early_stop_patience": early_stop_patience,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "learning_rate": learning_rate,
                    "optimizer_name": optimizer_name,
                    "seed": seed,
                    "save_best_only": save_best_only,
                    "freeze_backbone": freeze_backbone,
                    "device_name": device_name,
                    "run_id": run_id,
                },
                daemon=True,
                name="embedding-training",
            )
            self._task_thread.start()
        return self.runtime_status()

    def stop_training(self) -> dict[str, Any]:
        with self._task_lock:
            runtime = self.runtime_status()
            if not runtime.get("isRunning"):
                raise ValueError("当前没有运行中的训练任务")
            if not (self._task_thread and self._task_thread.is_alive()):
                runtime.update(
                    {
                        "isRunning": False,
                        "status": "interrupted",
                        "message": "训练线程不存在，状态已重置",
                        "updatedAt": self._now(),
                        "finishedAt": self._now(),
                    }
                )
                self._write_runtime(runtime)
                return runtime
            self._stop_event.set()
            runtime.update(
                {
                    "status": "stopping",
                    "message": "正在停止训练",
                    "updatedAt": self._now(),
                }
            )
            self._write_runtime(runtime)
            return runtime

    def _run_training_task(self, **kwargs) -> None:
        run_id = str(kwargs.get("run_id") or self.runtime_status().get("runId") or self._run_id())
        try:
            self.train(**kwargs)
        except TrainingStopped:
            runtime = self.runtime_status()
            runtime.update(
                {
                    "isRunning": False,
                    "status": "stopped",
                    "message": "训练已停止",
                    "updatedAt": self._now(),
                    "finishedAt": self._now(),
                    "stoppedByUser": True,
                }
            )
            self._write_runtime(runtime)
            self._archive_run_artifacts(run_id, self._read_metrics_payload(), runtime)
        except Exception as exc:
            runtime = self.runtime_status()
            runtime.update(
                {
                    "isRunning": False,
                    "status": "failed",
                    "message": str(exc),
                    "updatedAt": self._now(),
                    "finishedAt": self._now(),
                }
            )
            self._write_runtime(runtime)
            self._archive_run_artifacts(run_id, self._read_metrics_payload(), runtime)
        finally:
            with self._task_lock:
                self._task_thread = None
                self._stop_event.clear()

    def train(
        self,
        train_manifest_path: str,
        validation_manifest_path: str | None = None,
        epochs: int = 10,
        early_stop_patience: int = 3,
        batch_size: int = 16,
        num_workers: int = 2,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42,
        save_best_only: bool = True,
        freeze_backbone: bool = True,
        device_name: str = "auto",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        started_at = self.runtime_status().get("startedAt") or self._now()
        run_id = run_id or str(self.runtime_status().get("runId") or self._run_id())
        self._write_runtime(
            {
                "isRunning": True,
                "status": "running",
                "message": "正在读取训练数据",
                "currentEpoch": 0,
                "totalEpochs": epochs,
                "runId": run_id,
                "trainManifest": train_manifest_path,
                "validationManifest": validation_manifest_path,
                "device": device_name,
                "history": [],
                "startedAt": started_at,
                "updatedAt": self._now(),
                "finishedAt": None,
            }
        )
        try:
            device = self.resolve_device(device_name)
            self.device = device
            self._set_seed(seed)

            train_manifest_file = resolve_project_path(train_manifest_path)
            train_frame = read_csv(train_manifest_file)
            if train_frame.empty:
                raise ValueError("训练清单为空")
            validation_manifest_path = self._resolve_validation_manifest(train_manifest_path, validation_manifest_path)
            validation_manifest_file = resolve_project_path(validation_manifest_path) if validation_manifest_path else None
            validation_frame = read_csv(validation_manifest_file) if validation_manifest_file and validation_manifest_file.exists() else pd.DataFrame()
            class_names = self._class_names(train_frame, validation_frame)
            class_count = len(class_names)
            self._write_runtime(
                {
                    "isRunning": True,
                    "status": "running",
                    "message": "正在初始化模型",
                    "currentEpoch": 0,
                    "totalEpochs": epochs,
                    "runId": run_id,
                    "trainManifest": train_manifest_path,
                    "validationManifest": validation_manifest_path,
                    "device": device.type,
                    "history": [],
                    "startedAt": started_at,
                    "updatedAt": self._now(),
                    "finishedAt": None,
                }
            )

            backbone = ResNet101Backbone().to(device)
            embedder = SelfSimilarityEmbedding(target_dim=settings.feature_dim).to(device)
            arcface = ArcFaceHead(settings.feature_dim, class_count=class_count).to(device)

            if freeze_backbone:
                for parameter in backbone.parameters():
                    parameter.requires_grad = False

            train_records = train_frame.to_dict("records")
            validation_records = validation_frame.to_dict("records")
            train_dataset = ManifestImageDataset(train_records, backbone.transforms)
            loader_kwargs = self._loader_kwargs(device, num_workers)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **loader_kwargs)
            validation_loader = None
            validation_total = 0
            if validation_records:
                validation_dataset = ManifestImageDataset(validation_records, backbone.transforms)
                validation_total = len(validation_dataset)
                validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, **loader_kwargs)

            parameters = [parameter for parameter in list(backbone.parameters()) + list(embedder.parameters()) + list(arcface.parameters()) if parameter.requires_grad]
            if not parameters:
                raise ValueError("当前没有可训练参数")
            optimizer = self._build_optimizer(optimizer_name, parameters, learning_rate)
            criterion = nn.CrossEntropyLoss()

            history: list[dict[str, Any]] = []
            latest_confusion_matrix: list[list[int]] = []
            latest_per_class_metrics: list[dict[str, Any]] = []
            best_confusion_matrix: list[list[int]] = []
            best_per_class_metrics: list[dict[str, Any]] = []
            best_epoch = 0
            best_monitor_loss = float("inf")
            best_metrics: dict[str, Any] = {}
            patience_counter = 0
            stopped_early = False

            def make_epoch_record(
                epoch_number: int,
                train_metric: dict[str, Any] | None = None,
                validation_metric: dict[str, Any] | None = None,
                status: str = "训练中",
            ) -> dict[str, Any]:
                train_metric = train_metric or {}
                validation_metric = validation_metric or {}
                sample_count = train_metric.get("sampleCount")
                total_sample_count = train_metric.get("totalSampleCount")
                if total_sample_count and sample_count is not None and sample_count != total_sample_count:
                    sample_count = f"{sample_count}/{total_sample_count}"
                val_sample_count = validation_metric.get("sampleCount")
                val_total_count = validation_metric.get("totalSampleCount")
                if val_total_count and val_sample_count is not None and val_sample_count != val_total_count:
                    status = f"{status} {val_sample_count}/{val_total_count}"
                return {
                    "epoch": epoch_number,
                    "status": status,
                    "loss": train_metric.get("loss"),
                    "accuracy": train_metric.get("accuracy"),
                    "precision": train_metric.get("precision"),
                    "recall": train_metric.get("recall"),
                    "macroF1": train_metric.get("macroF1"),
                    "sampleCount": sample_count,
                    "valLoss": validation_metric.get("loss"),
                    "valAccuracy": validation_metric.get("accuracy"),
                    "valPrecision": validation_metric.get("precision"),
                    "valRecall": validation_metric.get("recall"),
                    "valMacroF1": validation_metric.get("macroF1"),
                    "learningRate": learning_rate,
                }

            def write_runtime_progress(message: str, epoch_number: int, runtime_history: list[dict[str, Any]]) -> None:
                self._write_runtime(
                    {
                        "isRunning": True,
                        "status": "running",
                        "message": message,
                        "currentEpoch": epoch_number,
                        "totalEpochs": epochs,
                        "runId": run_id,
                        "trainManifest": train_manifest_path,
                        "validationManifest": validation_manifest_path,
                        "device": device.type,
                        "history": runtime_history,
                        "startedAt": started_at,
                        "updatedAt": self._now(),
                        "finishedAt": None,
                    }
                )

            for epoch in range(epochs):
                self._raise_if_stopped()
                epoch_number = epoch + 1
                write_runtime_progress(
                    f"第 {epoch_number}/{epochs} 轮训练中",
                    epoch_number,
                    [*history, make_epoch_record(epoch_number, status="训练中")],
                )

                def write_train_progress(partial_metrics: dict[str, Any]) -> None:
                    write_runtime_progress(
                        f"第 {epoch_number}/{epochs} 轮训练中",
                        epoch_number,
                        [*history, make_epoch_record(epoch_number, partial_metrics, status="训练中")],
                    )

                train_metrics = self._run_epoch(
                    loader=train_loader,
                    backbone=backbone,
                    embedder=embedder,
                    arcface=arcface,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    class_names=class_names,
                    freeze_backbone=freeze_backbone,
                    progress_callback=write_train_progress,
                )
                write_runtime_progress(
                    f"第 {epoch_number}/{epochs} 轮验证中" if validation_loader is not None else f"第 {epoch_number}/{epochs} 轮训练中",
                    epoch_number,
                    [*history, make_epoch_record(epoch_number, train_metrics, status="验证中" if validation_loader is not None else "训练中")],
                )
                validation_metrics = None
                if validation_loader is not None:
                    self._raise_if_stopped()

                    def write_validation_progress(partial_metrics: dict[str, Any]) -> None:
                        write_runtime_progress(
                            f"第 {epoch_number}/{epochs} 轮验证中",
                            epoch_number,
                            [*history, make_epoch_record(epoch_number, train_metrics, partial_metrics, status="验证中")],
                        )

                    validation_metrics = self._run_epoch(
                        loader=validation_loader,
                        backbone=backbone,
                        embedder=embedder,
                        arcface=arcface,
                        criterion=criterion,
                        optimizer=None,
                        device=device,
                        class_names=class_names,
                        freeze_backbone=freeze_backbone,
                        progress_callback=write_validation_progress,
                    )

                epoch_record = {
                    "epoch": epoch_number,
                    "status": "完成",
                    "loss": train_metrics["loss"],
                    "accuracy": train_metrics["accuracy"],
                    "precision": train_metrics["precision"],
                    "recall": train_metrics["recall"],
                    "macroF1": train_metrics["macroF1"],
                    "sampleCount": train_metrics["sampleCount"],
                    "valLoss": validation_metrics["loss"] if validation_metrics else None,
                    "valAccuracy": validation_metrics["accuracy"] if validation_metrics else None,
                    "valPrecision": validation_metrics["precision"] if validation_metrics else None,
                    "valRecall": validation_metrics["recall"] if validation_metrics else None,
                    "valMacroF1": validation_metrics["macroF1"] if validation_metrics else None,
                    "learningRate": learning_rate,
                }
                history.append(epoch_record)
                self._raise_if_stopped()

                epoch_snapshot = validation_metrics or train_metrics
                latest_confusion_matrix = epoch_snapshot["confusionMatrix"]
                latest_per_class_metrics = epoch_snapshot["perClassMetrics"]
                monitor_loss = validation_metrics["loss"] if validation_metrics else train_metrics["loss"]
                improved = monitor_loss < best_monitor_loss - 1e-8

                if improved:
                    best_monitor_loss = monitor_loss
                    best_epoch = epoch_number
                    best_metrics = {
                        "loss": epoch_record["loss"],
                        "accuracy": epoch_record["accuracy"],
                        "precision": epoch_record["precision"],
                        "recall": epoch_record["recall"],
                        "macroF1": epoch_record["macroF1"],
                        "valLoss": epoch_record["valLoss"],
                        "valAccuracy": epoch_record["valAccuracy"],
                        "valPrecision": epoch_record["valPrecision"],
                        "valRecall": epoch_record["valRecall"],
                        "valMacroF1": epoch_record["valMacroF1"],
                    }
                    best_confusion_matrix = latest_confusion_matrix
                    best_per_class_metrics = latest_per_class_metrics
                    patience_counter = 0
                    self._save_checkpoint(
                        self.checkpoint_root / "embedding_latest.pt",
                        backbone,
                        embedder,
                        arcface,
                        history,
                        class_names,
                        device,
                        best_epoch,
                    )
                    self._save_checkpoint(
                        self.checkpoint_root / "embedding_best.pt",
                        backbone,
                        embedder,
                        arcface,
                        history,
                        class_names,
                        device,
                        best_epoch,
                    )
                else:
                    patience_counter += 1
                    if not save_best_only:
                        self._save_checkpoint(
                            self.checkpoint_root / "embedding_latest.pt",
                            backbone,
                            embedder,
                            arcface,
                            history,
                            class_names,
                            device,
                            best_epoch,
                        )

                metrics_payload = {
                    "runId": run_id,
                    "device": device.type,
                    "trainManifest": train_manifest_path,
                    "validationManifest": validation_manifest_path,
                    "epochs": epochs,
                    "earlyStopPatience": early_stop_patience,
                    "batchSize": batch_size,
                    "numWorkers": num_workers,
                    "optimizer": optimizer_name,
                    "learningRate": learning_rate,
                    "seed": seed,
                    "saveBestOnly": save_best_only,
                    "freezeBackbone": freeze_backbone,
                    "architecture": embedder.architecture_name,
                    "embeddingConfig": embedder.config(),
                    "classNames": class_names,
                    "history": history,
                    "bestEpoch": best_epoch,
                    "bestMonitorLoss": None if best_monitor_loss == float("inf") else round(best_monitor_loss, 6),
                    "bestMetrics": best_metrics,
                    "bestConfusionMatrix": best_confusion_matrix,
                    "bestPerClassMetrics": best_per_class_metrics,
                    "latestConfusionMatrix": latest_confusion_matrix,
                    "latestPerClassMetrics": latest_per_class_metrics,
                    "stoppedEarly": False,
                }
                self._write_metrics_payload(metrics_payload)
                self._write_runtime(
                    {
                        "isRunning": True,
                        "status": "running",
                        "message": f"第 {epoch_number}/{epochs} 轮已完成",
                        "currentEpoch": epoch_number,
                        "totalEpochs": epochs,
                        "runId": run_id,
                        "trainManifest": train_manifest_path,
                        "validationManifest": validation_manifest_path,
                        "device": device.type,
                        "history": history,
                        "startedAt": started_at,
                        "updatedAt": self._now(),
                        "finishedAt": None,
                    }
                )

                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    stopped_early = True
                    break

            if not save_best_only and history:
                self._save_checkpoint(
                    self.checkpoint_root / "embedding_latest.pt",
                    backbone,
                    embedder,
                    arcface,
                    history,
                    class_names,
                    device,
                    best_epoch,
                )

            final_payload = {
                **metrics_payload,
                "stoppedEarly": stopped_early,
            }
            self._write_metrics_payload(final_payload)
            completed_runtime = {
                "isRunning": False,
                "status": "completed",
                "message": "训练完成",
                "currentEpoch": len(history),
                "totalEpochs": epochs,
                "runId": run_id,
                "trainManifest": train_manifest_path,
                "validationManifest": validation_manifest_path,
                "device": device.type,
                "history": history,
                "startedAt": started_at,
                "updatedAt": self._now(),
                "finishedAt": self._now(),
                "stoppedEarly": stopped_early,
            }
            self._write_runtime(completed_runtime)
            self._archive_run_artifacts(run_id, final_payload, completed_runtime)
            return self._serialize_for_output(
                {
                    "checkpointPath": str(self.checkpoint_root / "embedding_best.pt"),
                    "metricsPath": str(self.history_path),
                    "history": history,
                    "device": device.type,
                    "bestEpoch": best_epoch,
                    "stoppedEarly": stopped_early,
                }
            )
        except TrainingStopped:
            raise
        except Exception as exc:
            failed_runtime = {
                "isRunning": False,
                "status": "failed",
                "message": str(exc),
                "currentEpoch": self.runtime_status().get("currentEpoch", 0),
                "totalEpochs": epochs,
                "runId": run_id,
                "trainManifest": train_manifest_path,
                "validationManifest": validation_manifest_path,
                "device": device_name,
                "history": self.runtime_status().get("history", []),
                "startedAt": started_at,
                "updatedAt": self._now(),
                "finishedAt": self._now(),
            }
            self._write_runtime(failed_runtime)
            raise

    def extract_features(
        self,
        manifest_path: str,
        output_manifest_name: str,
        checkpoint_path: str | None = None,
        mode: str = "embedding",
        model_name: str | None = None,
        resume: bool = True,
        flush_every: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        device = self.resolve_device("auto")
        self.device = device
        backbone = ResNet101Backbone().to(device).eval()
        embedder = SelfSimilarityEmbedding(target_dim=settings.feature_dim).to(device).eval()
        checkpoint_metadata: dict[str, Any] = {}

        if mode == "embedding" and not checkpoint_path:
            raise ValueError("请选择模型权重")
        if checkpoint_path:
            checkpoint_metadata = self._load_embedding_checkpoint(checkpoint_path, backbone, embedder, device)

        records = read_csv(manifest_path).to_dict("records")
        feature_dir = self.feature_manifest_root / Path(output_manifest_name).stem
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_manifest_path = self.feature_manifest_root / output_manifest_name

        exported: list[dict[str, Any]] = []
        skipped_existing = 0
        feature_metadata: dict[str, Any] = {"feature_mode": mode}
        if mode == "embedding":
            feature_metadata["architecture"] = embedder.architecture_name
            for key in ("checkpoint_mtime_ns", "checkpoint_size_bytes"):
                if key in checkpoint_metadata:
                    feature_metadata[key] = checkpoint_metadata[key]
        if checkpoint_path:
            feature_metadata["checkpoint_path"] = serialize_project_path(resolve_project_path(checkpoint_path))
        if model_name:
            feature_metadata["model_name"] = model_name

        def flush_manifest() -> None:
            if exported:
                write_csv(output_manifest_path, pd.DataFrame(exported))

        total_records = len(records)
        for index, record in enumerate(records, start=1):
            feature_path = feature_dir / f"{record['image_id']}.npy"
            if resume and feature_path.exists():
                exported.append({**record, "feature_path": serialize_project_path(feature_path), **feature_metadata})
                skipped_existing += 1
                if flush_every > 0 and index % flush_every == 0:
                    flush_manifest()
                    if progress_callback:
                        progress_callback(index, total_records)
                continue

            image = Image.open(resolve_project_path(record["file_path"])).convert("RGB")
            tensor = backbone.transforms(image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            with torch.inference_mode():
                feature_map, pooled = backbone.forward_features(tensor)
                if mode == "baseline":
                    vector = pooled.squeeze(0).cpu().numpy()
                else:
                    vector = embedder(feature_map).squeeze(0).cpu().numpy()
            np.save(feature_path, vector.astype("float32"))
            exported.append({**record, "feature_path": serialize_project_path(feature_path), **feature_metadata})
            if flush_every > 0 and index % flush_every == 0:
                flush_manifest()
                if progress_callback:
                    progress_callback(index, total_records)

        flush_manifest()
        if progress_callback:
            progress_callback(total_records, total_records)
        return self._serialize_for_output(
            {
                "manifestPath": str(output_manifest_path),
                "count": len(exported),
                "mode": mode,
                "resumed": resume,
                "skippedExisting": skipped_existing,
                "computed": len(exported) - skipped_existing,
            }
        )

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        backbone: ResNet101Backbone,
        embedder: SelfSimilarityEmbedding,
        arcface: ArcFaceHead,
        history: list[dict[str, Any]],
        class_names: list[str],
        device: torch.device,
        best_epoch: int,
    ) -> None:
        torch.save(
            {
                "backbone": backbone.state_dict(),
                "embedder": embedder.state_dict(),
                "arcface": arcface.state_dict(),
                "history": history,
                "architecture": embedder.architecture_name,
                "embedding_config": embedder.config(),
                "feature_dim": settings.feature_dim,
                "class_names": class_names,
                "device": device.type,
                "best_epoch": best_epoch,
            },
            checkpoint_path,
        )

    def _load_embedding_checkpoint(
        self,
        checkpoint_path: str,
        backbone: ResNet101Backbone,
        embedder: SelfSimilarityEmbedding,
        device: torch.device,
    ) -> dict[str, Any]:
        resolved_checkpoint_path = resolve_project_path(checkpoint_path)
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
        architecture = self._checkpoint_architecture(checkpoint, embedder)
        if architecture != embedder.architecture_name:
            raise ValueError("模型权重结构已更新，请重新训练自相似嵌入模型")
        if int(checkpoint.get("feature_dim", settings.feature_dim)) != settings.feature_dim:
            raise ValueError("模型权重特征维度与当前配置不一致")
        backbone.load_state_dict(checkpoint["backbone"], strict=False)
        embedder.load_state_dict(checkpoint["embedder"], strict=True)
        return {
            "architecture": architecture,
            "feature_dim": int(checkpoint.get("feature_dim", settings.feature_dim)),
            "checkpoint_mtime_ns": str(resolved_checkpoint_path.stat().st_mtime_ns),
            "checkpoint_size_bytes": str(resolved_checkpoint_path.stat().st_size),
        }

    def embedding_checkpoint_metadata(self, checkpoint_path: str) -> dict[str, Any]:
        resolved_checkpoint_path = resolve_project_path(checkpoint_path)
        checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
        probe = SelfSimilarityEmbedding(target_dim=settings.feature_dim)
        architecture = self._checkpoint_architecture(checkpoint, probe)
        if architecture != probe.architecture_name:
            raise ValueError("模型权重结构已更新，请重新训练自相似嵌入模型")
        return {
            "architecture": architecture,
            "feature_dim": int(checkpoint.get("feature_dim", settings.feature_dim)),
            "checkpoint_mtime_ns": str(resolved_checkpoint_path.stat().st_mtime_ns),
            "checkpoint_size_bytes": str(resolved_checkpoint_path.stat().st_size),
        }

    @staticmethod
    def _checkpoint_architecture(checkpoint: dict[str, Any], embedder: SelfSimilarityEmbedding) -> str | None:
        architecture = checkpoint.get("architecture")
        if architecture:
            return str(architecture)
        keys = set(checkpoint.get("embedder", {}).keys())
        if any(key.startswith(("tensor_encoder.", "feature_fusion.", "input_projection.", "initial_norm.")) for key in keys):
            return embedder.architecture_name
        if any(key.startswith("projection.") for key in keys):
            return "legacy_projection_embedding"
        return None

    def _write_metrics_payload(self, payload: dict[str, Any]) -> None:
        self.history_path.write_text(
            json.dumps(self._serialize_for_output(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_runtime(self, payload: dict[str, Any]) -> None:
        self.runtime_path.write_text(
            json.dumps(self._serialize_for_output(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_metrics_payload(self) -> dict[str, Any] | None:
        if not self.history_path.exists():
            return None
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _archive_run_artifacts(
        self,
        run_id: str,
        metrics_payload: dict[str, Any] | None,
        runtime_payload: dict[str, Any] | None,
    ) -> None:
        run_dir = self.training_runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if metrics_payload is not None:
            (run_dir / "metrics.json").write_text(
                json.dumps(self._serialize_for_output(metrics_payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if runtime_payload is not None:
            (run_dir / "runtime.json").write_text(
                json.dumps(self._serialize_for_output(runtime_payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        summary_payload = {
            "runId": run_id,
            "status": runtime_payload.get("status") if runtime_payload else None,
            "device": runtime_payload.get("device") if runtime_payload else None,
            "trainManifest": runtime_payload.get("trainManifest") if runtime_payload else None,
            "validationManifest": runtime_payload.get("validationManifest") if runtime_payload else None,
            "bestEpoch": metrics_payload.get("bestEpoch") if metrics_payload else None,
            "bestMetrics": metrics_payload.get("bestMetrics") if metrics_payload else None,
            "startedAt": runtime_payload.get("startedAt") if runtime_payload else None,
            "finishedAt": runtime_payload.get("finishedAt") if runtime_payload else None,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(self._serialize_for_output(summary_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for checkpoint_name in ("embedding_best.pt", "embedding_latest.pt"):
            source = self.checkpoint_root / checkpoint_name
            if source.exists():
                shutil.copy2(source, run_dir / checkpoint_name)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _run_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _serialize_for_output(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: EmbeddingTrainingService._serialize_for_output(item) for key, item in value.items()}
        if isinstance(value, list):
            return [EmbeddingTrainingService._serialize_for_output(item) for item in value]
        if isinstance(value, str):
            return serialize_project_path(value)
        return value

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_optimizer(name: str, parameters: list[torch.nn.Parameter], learning_rate: float):
        optimizer_name = name.strip().lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate)
        if optimizer_name == "sgd":
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
        raise ValueError("优化器不存在")

    def _resolve_validation_manifest(self, train_manifest_path: str, validation_manifest_path: str | None) -> str | None:
        if validation_manifest_path:
            return serialize_project_path(validation_manifest_path)
        candidate = resolve_project_path(train_manifest_path)
        if "train" in candidate.name:
            sibling = candidate.with_name(candidate.name.replace("train", "test", 1))
            if sibling.exists():
                return serialize_project_path(sibling)
        return None

    @staticmethod
    def _loader_kwargs(device: torch.device, num_workers: int) -> dict[str, Any]:
        if device.type != "cuda":
            return {}
        options: dict[str, Any] = {"pin_memory": True}
        if num_workers > 0:
            options["persistent_workers"] = True
        return options

    @staticmethod
    def _class_names(*frames: pd.DataFrame) -> list[str]:
        mapping: dict[int, str] = {}
        for frame in frames:
            if frame is None or frame.empty:
                continue
            if "label_index" not in frame.columns or "label_name" not in frame.columns:
                continue
            dedup = frame[["label_index", "label_name"]].drop_duplicates().sort_values("label_index")
            for row in dedup.itertuples(index=False):
                mapping[int(row.label_index)] = str(row.label_name)
        if not mapping:
            return []
        max_index = max(mapping)
        return [mapping.get(index, f"class_{index}") for index in range(max_index + 1)]

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise TrainingStopped("训练已停止")

    def _run_epoch(
        self,
        loader: DataLoader,
        backbone: ResNet101Backbone,
        embedder: SelfSimilarityEmbedding,
        arcface: ArcFaceHead,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        class_names: list[str],
        freeze_backbone: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        training = optimizer is not None
        if training:
            backbone.train()
            if freeze_backbone:
                backbone.stem.eval()
                backbone.layer1.eval()
                backbone.layer2.eval()
                backbone.layer3.eval()
                backbone.layer4.eval()
        else:
            backbone.eval()
        embedder.train(training)
        arcface.train(training)

        total_loss = 0.0
        total_count = 0
        predictions: list[int] = []
        targets: list[int] = []
        dataset_total = len(loader.dataset) if hasattr(loader, "dataset") else 0
        progress_interval = max(1, min(50, max(len(loader) // 20, 1)))

        for batch_index, (images, labels) in enumerate(loader, start=1):
            self._raise_if_stopped()
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training and not freeze_backbone):
                feature_map, _ = backbone.forward_features(images)
            if freeze_backbone:
                feature_map = feature_map.detach()
            with torch.set_grad_enabled(training):
                embeddings = embedder(feature_map)
                logits = arcface(embeddings, labels)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()

            preds = logits.argmax(dim=1)
            predictions.extend(preds.detach().cpu().numpy().tolist())
            targets.extend(labels.detach().cpu().numpy().tolist())
            total_loss += float(loss.item()) * labels.size(0)
            total_count += int(labels.size(0))
            if progress_callback and (
                batch_index == 1 or batch_index % progress_interval == 0 or (dataset_total and total_count >= dataset_total)
            ):
                partial_metrics = self._classification_metrics(np.array(targets), np.array(predictions), class_names)
                partial_metrics.update(
                    {
                        "loss": round(total_loss / max(total_count, 1), 6),
                        "sampleCount": total_count,
                        "totalSampleCount": dataset_total,
                    }
                )
                progress_callback(partial_metrics)

        metrics = self._classification_metrics(np.array(targets), np.array(predictions), class_names)
        metrics.update(
            {
                "loss": round(total_loss / max(total_count, 1), 6),
                "sampleCount": total_count,
                "totalSampleCount": dataset_total,
            }
        )
        return metrics

    @staticmethod
    def _classification_metrics(targets: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict[str, Any]:
        class_count = len(class_names)
        confusion = np.zeros((class_count, class_count), dtype=int)
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            if 0 <= target < class_count and 0 <= prediction < class_count:
                confusion[target, prediction] += 1

        total = int(confusion.sum())
        accuracy = float(np.trace(confusion) / total) if total else 0.0
        class_rows: list[dict[str, Any]] = []
        for index, label_name in enumerate(class_names):
            true_positive = int(confusion[index, index])
            false_positive = int(confusion[:, index].sum() - true_positive)
            false_negative = int(confusion[index, :].sum() - true_positive)
            support = int(confusion[index, :].sum())
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            class_rows.append(
                {
                    "labelIndex": index,
                    "labelName": label_name,
                    "support": support,
                    "precision": round(float(precision), 6),
                    "recall": round(float(recall), 6),
                    "f1": round(float(f1_score), 6),
                }
            )

        precision_macro = float(np.mean([row["precision"] for row in class_rows])) if class_rows else 0.0
        recall_macro = float(np.mean([row["recall"] for row in class_rows])) if class_rows else 0.0
        f1_macro = float(np.mean([row["f1"] for row in class_rows])) if class_rows else 0.0
        return {
            "accuracy": round(accuracy, 6),
            "precision": round(precision_macro, 6),
            "recall": round(recall_macro, 6),
            "macroF1": round(f1_macro, 6),
            "confusionMatrix": confusion.tolist(),
            "perClassMetrics": class_rows,
        }
