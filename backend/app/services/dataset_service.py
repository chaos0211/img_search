from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torchvision.datasets import CIFAR10

from backend.app.config import settings
from backend.app.utils.file_utils import read_csv, resolve_project_path, serialize_project_path, write_csv, write_json


@dataclass
class DatasetPaths:
    raw_root: Path
    processed_root: Path
    gallery_root: Path
    query_root: Path
    manifests_root: Path


class DatasetService:
    def __init__(self):
        self.project_root = settings.project_root
        self.paths = DatasetPaths(
            raw_root=self.project_root / "data" / "raw" / "cifar10",
            processed_root=self.project_root / "data" / "processed" / "cifar10",
            gallery_root=self.project_root / "data" / "gallery" / "cifar10",
            query_root=self.project_root / "data" / "query" / "cifar10",
            manifests_root=self.project_root / "data" / "manifests",
        )
        for path in (
            self.paths.raw_root,
            self.paths.processed_root,
            self.paths.gallery_root,
            self.paths.query_root,
            self.paths.manifests_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def prepare_cifar10(
        self,
        train_limit: int | None = None,
        test_limit: int | None = None,
        gallery_per_class: int = 200,
        query_per_class: int = 40,
        clean_existing: bool = False,
    ) -> dict[str, Any]:
        if clean_existing:
            for path in (self.paths.processed_root, self.paths.gallery_root, self.paths.query_root):
                if path.exists():
                    shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean",
                category=Warning,
            )
            train_set = CIFAR10(root=str(self.paths.raw_root), train=True, download=True)
            test_set = CIFAR10(root=str(self.paths.raw_root), train=False, download=True)

        train_records = self._export_split(train_set, "train", train_limit)
        test_records = self._export_split(test_set, "test", test_limit)

        gallery_records = self._materialize_partition(train_records, self.paths.gallery_root, "gallery", gallery_per_class)
        query_records = self._materialize_partition(test_records, self.paths.query_root, "query", query_per_class)

        train_manifest = self.paths.manifests_root / "cifar10_train.csv"
        test_manifest = self.paths.manifests_root / "cifar10_test.csv"
        gallery_manifest = self.paths.manifests_root / "cifar10_gallery.csv"
        query_manifest = self.paths.manifests_root / "cifar10_query.csv"

        write_csv(train_manifest, pd.DataFrame(train_records))
        write_csv(test_manifest, pd.DataFrame(test_records))
        write_csv(gallery_manifest, pd.DataFrame(gallery_records))
        write_csv(query_manifest, pd.DataFrame(query_records))

        summary = {
            "trainCount": len(train_records),
            "testCount": len(test_records),
            "galleryCount": len(gallery_records),
            "queryCount": len(query_records),
            "classes": sorted({record["label_name"] for record in train_records}),
            "trainManifest": self._relative_path(train_manifest),
            "testManifest": self._relative_path(test_manifest),
            "galleryManifest": self._relative_path(gallery_manifest),
            "queryManifest": self._relative_path(query_manifest),
        }
        write_json(self.paths.manifests_root / "cifar10_summary.json", summary)
        return summary

    def load_manifest(self, manifest_name: str) -> pd.DataFrame:
        path = self.paths.manifests_root / manifest_name
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        return read_csv(path)

    def _export_split(self, dataset: CIFAR10, split_name: str, limit: int | None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {name: 0 for name in dataset.classes}
        max_count = min(limit, len(dataset)) if limit else len(dataset)

        for index in range(max_count):
            image, label_index = dataset[index]
            label_name = dataset.classes[label_index]
            class_counts[label_name] += 1
            output_dir = self.paths.processed_root / split_name / label_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{split_name}_{index:05d}.png"
            verified = image.convert("RGB")
            verified.save(output_path)
            with output_path.open("rb") as handle:
                sha1 = hashlib.sha1(handle.read()).hexdigest()
            records.append(
                {
                    "image_id": f"{split_name}_{index:05d}",
                    "split_name": split_name,
                    "label_index": label_index,
                    "label_name": label_name,
                        "file_path": serialize_project_path(output_path),
                    "width": verified.width,
                    "height": verified.height,
                    "sha1": sha1,
                    "is_valid": 1,
                }
            )
        return records

    def _materialize_partition(
        self,
        records: list[dict[str, Any]],
        target_root: Path,
        partition_name: str,
        per_class_limit: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["label_name"], []).append(record)

        selected_records: list[dict[str, Any]] = []
        for label_name, label_records in grouped.items():
            for record in label_records[:per_class_limit]:
                source_path = resolve_project_path(record["file_path"])
                target_dir = target_root / label_name
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / source_path.name
                if not target_path.exists():
                    shutil.copy2(source_path, target_path)
                selected_records.append({**record, "partition": partition_name, "file_path": serialize_project_path(target_path)})
        return selected_records

    def _relative_path(self, path: Path) -> str:
        return serialize_project_path(path)
