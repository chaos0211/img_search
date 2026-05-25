from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.database import get_connection, init_database
from backend.app.services.feature_service import get_feature_service
from backend.app.services.vector_index_service import VectorIndexService
from backend.app.utils.file_utils import serialize_project_path


DEFAULT_MANIFEST = settings.project_root / "data" / "manifests" / "cifar10_test.csv"
FEATURE_MODEL_NAME = "resnet101_imagenet_pooling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset gallery data from a fixed CIFAR-10 manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--created-by", type=int, default=1)
    parser.add_argument("--source", default="cifar10_test")
    parser.add_argument("--progress-step", type=int, default=100)
    return parser.parse_args()


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_gallery_runtime_data() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM duplicate_actions")
            cursor.execute("DELETE FROM retrieval_logs")
            cursor.execute("DELETE FROM cluster_runs")
            cursor.execute("DELETE FROM images")
            for table_name in ("duplicate_actions", "retrieval_logs", "cluster_runs", "images"):
                cursor.execute(f"ALTER TABLE {table_name} AUTO_INCREMENT = 1")

    clear_directory(settings.gallery_root)
    clear_directory(settings.thumbnail_root)
    clear_directory(settings.feature_root)
    clear_directory(settings.index_root / "faiss")


def load_manifest_rows(manifest_path: Path, limit: int) -> list[dict[str, str]]:
    resolved_manifest = manifest_path if manifest_path.is_absolute() else settings.project_root / manifest_path
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"manifest not found: {resolved_manifest}")

    rows: list[dict[str, str]] = []
    with resolved_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("is_valid", "1")) != "1":
                continue
            source_path = settings.project_root / str(row["file_path"])
            if not source_path.exists():
                raise FileNotFoundError(f"image not found: {source_path}")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def verify_sha1(source_path: Path, expected_sha1: str) -> None:
    if not expected_sha1:
        return
    actual_sha1 = hashlib.sha1(source_path.read_bytes()).hexdigest()
    if actual_sha1 != expected_sha1:
        raise ValueError(f"sha1 mismatch: {source_path}")


def save_thumbnail(source_path: Path, thumbnail_path: Path) -> tuple[int, int, str]:
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        thumbnail = rgb_image.copy()
        thumbnail.thumbnail((320, 320))
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail.save(thumbnail_path, format="JPEG", quality=88)
        mime_type = Image.MIME.get(image.format or "PNG", "image/png")
    return width, height, mime_type


def insert_batch(records: list[tuple]) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO images (
                    original_name, stored_name, file_path, thumbnail_path, feature_path, source,
                    label_name, split_name, width, height, mime_type, feature_model, feature_dim, created_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )


def import_gallery(rows: list[dict[str, str]], source_name: str, created_by: int | None, progress_step: int) -> None:
    feature_service = get_feature_service()
    pending_records: list[tuple] = []
    total = len(rows)

    for index, row in enumerate(rows, start=1):
        source_path = settings.project_root / str(row["file_path"])
        verify_sha1(source_path, str(row.get("sha1") or ""))

        image_id = str(row["image_id"])
        suffix = source_path.suffix.lower() or ".png"
        stored_name = f"{image_id}{suffix}"
        stored_path = settings.gallery_root / stored_name
        thumbnail_path = settings.thumbnail_root / f"{image_id}.jpg"
        feature_path = settings.feature_root / f"{image_id}.npy"

        shutil.copy2(source_path, stored_path)
        width, height, mime_type = save_thumbnail(stored_path, thumbnail_path)
        feature = feature_service.extract(str(stored_path), mode="baseline").astype(np.float32)
        np.save(feature_path, feature)

        pending_records.append(
            (
                stored_name,
                stored_name,
                serialize_project_path(stored_path),
                serialize_project_path(thumbnail_path),
                serialize_project_path(feature_path),
                source_name,
                str(row["label_name"]),
                str(row.get("split_name") or "test"),
                width,
                height,
                mime_type,
                FEATURE_MODEL_NAME,
                int(feature.shape[0]),
                created_by,
            )
        )

        if len(pending_records) >= 100:
            insert_batch(pending_records)
            pending_records.clear()

        if index == total or index % progress_step == 0:
            print(f"Imported {index}/{total}", flush=True)

    if pending_records:
        insert_batch(pending_records)


def print_verification() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(DISTINCT original_name) AS distinct_names,
                    COUNT(DISTINCT CONCAT(original_name, '|', label_name)) AS distinct_name_labels,
                    COUNT(DISTINCT feature_path) AS distinct_features
                FROM images
                WHERE is_deleted = 0
                """
            )
            summary = cursor.fetchone()
            cursor.execute(
                """
                SELECT label_name, COUNT(*) AS total
                FROM images
                WHERE is_deleted = 0
                GROUP BY label_name
                ORDER BY label_name ASC
                """
            )
            distribution = cursor.fetchall()

    print("Gallery verification", flush=True)
    print(summary, flush=True)
    print(distribution, flush=True)


def main() -> None:
    args = parse_args()
    init_database()
    rows = load_manifest_rows(args.manifest, args.limit)
    if not rows:
        raise ValueError("manifest has no valid rows")

    print(f"Resetting gallery, target rows: {len(rows)}", flush=True)
    clear_gallery_runtime_data()
    import_gallery(rows, args.source, args.created_by, args.progress_step)
    index_status = VectorIndexService().rebuild_gallery_index()
    print_verification()
    print("FAISS status", flush=True)
    print(index_status, flush=True)


if __name__ == "__main__":
    main()
