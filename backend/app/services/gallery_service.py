from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from backend.app.config import settings
from backend.app.database import execute, fetch_all, fetch_one
from backend.app.services.feature_service import get_feature_service
from backend.app.utils.file_utils import read_csv, resolve_project_path, serialize_project_path
from backend.app.utils.image_utils import save_base64_image, thumbnail_to_data_url
from backend.app.utils.pagination import paginate


class GalleryService:
    UNSET_LABEL_VALUE = "__unset__"
    UNSET_LABEL_NAME = "未设置"

    def ensure_test_batches(self) -> None:
        row = fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM images
            WHERE is_deleted = 0 AND source = 'cifar10_test' AND split_name = 'test' AND original_name LIKE 'test\\_%'
            """
        )
        if not int((row or {}).get("total") or 0):
            return
        execute(
            """
            UPDATE images
            SET source = 'test_set',
                split_name = CONCAT(
                    'test',
                    FLOOR(CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(original_name, '_', -1), '.', 1) AS UNSIGNED) / 100) + 1
                )
            WHERE is_deleted = 0 AND source = 'cifar10_test' AND split_name = 'test' AND original_name LIKE 'test\\_%'
            """
        )

    def create_from_upload(
        self,
        data_url: str,
        original_name: str,
        creator_id: int | None,
        source: str = "upload",
        label_name: str | None = None,
        split_name: str | None = None,
    ) -> dict:
        label_name = self._normalize_label_name(label_name)
        saved = save_base64_image(data_url, original_name, settings.gallery_root)
        feature = get_feature_service().extract(str(saved["file_path"]), mode="baseline")
        feature_name = f"{Path(str(saved['stored_name'])).stem}.npy"
        feature_path = settings.feature_root / feature_name
        np.save(feature_path, feature)

        image_id = execute(
            """
            INSERT INTO images (
                original_name, stored_name, file_path, thumbnail_path, feature_path, source,
                label_name, split_name, width, height, mime_type, feature_model, feature_dim, created_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                original_name,
                saved["stored_name"],
                saved["file_path"],
                saved["thumbnail_path"],
                serialize_project_path(feature_path),
                source,
                label_name,
                split_name,
                saved["width"],
                saved["height"],
                saved["mime_type"],
                "resnet101_imagenet_pooling",
                int(feature.shape[0]),
                creator_id,
            ),
        )
        return self.get_image(int(image_id))

    def create_from_file(
        self,
        image_path: str | Path,
        original_name: str,
        creator_id: int | None,
        source: str,
        label_name: str | None = None,
        split_name: str | None = None,
    ) -> dict:
        label_name = self._normalize_label_name(label_name)
        source_path = resolve_project_path(image_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        suffix = source_path.suffix.lower()
        if suffix not in settings.allowed_image_suffixes:
            suffix = ".png"
        stored_name = f"{uuid.uuid4().hex}{suffix}"
        stored_path = settings.gallery_root / stored_name
        stored_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(source_path).convert("RGB")
        image.save(stored_path)

        thumbnail_path = settings.thumbnail_root / f"{stored_path.stem}.jpg"
        thumbnail = image.copy()
        thumbnail.thumbnail((320, 320))
        thumbnail.save(thumbnail_path, format="JPEG", quality=88)

        feature = get_feature_service().extract(str(stored_path), mode="baseline")
        feature_path = settings.feature_root / f"{stored_path.stem}.npy"
        np.save(feature_path, feature.astype(np.float32))

        image_id = execute(
            """
            INSERT INTO images (
                original_name, stored_name, file_path, thumbnail_path, feature_path, source,
                label_name, split_name, width, height, mime_type, feature_model, feature_dim, created_by
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                original_name,
                stored_name,
                serialize_project_path(stored_path),
                serialize_project_path(thumbnail_path),
                serialize_project_path(feature_path),
                source,
                label_name,
                split_name,
                image.width,
                image.height,
                mimetypes.guess_type(original_name)[0] or "image/png",
                "resnet101_imagenet_pooling",
                int(feature.shape[0]),
                creator_id,
            ),
        )
        return self.get_image(int(image_id))

    def get_image(self, image_id: int) -> dict:
        row = fetch_one(
            """
            SELECT images.*, users.display_name AS creator_name
            FROM images
            LEFT JOIN users ON users.id = images.created_by
            WHERE images.id = %s AND images.is_deleted = 0
            LIMIT 1
            """,
            (image_id,),
        )
        if not row:
            raise ValueError("图片不存在")
        return self._serialize_image(row)

    def list_images(self) -> list[dict]:
        rows = fetch_all(
            """
            SELECT images.*, users.display_name AS creator_name
            FROM images
            LEFT JOIN users ON users.id = images.created_by
            WHERE images.is_deleted = 0
            ORDER BY images.id DESC
            """
        )
        return [self._serialize_image(row) for row in rows]

    def list_images_page(self, page: int, page_size: int, batch_key: str = "", label_name: str = "") -> dict:
        where_sql, params = self._gallery_where(batch_key=batch_key, label_name=label_name)
        total_row = fetch_one(f"SELECT COUNT(*) AS total FROM images WHERE {where_sql}", tuple(params))
        total = int((total_row or {}).get("total") or 0)
        paged = paginate(list(range(total)), page, page_size)
        offset = (paged["page"] - 1) * page_size
        alias_where_sql, alias_params = self._gallery_where(alias="images", batch_key=batch_key, label_name=label_name)
        rows = fetch_all(
            f"""
            SELECT images.*, users.display_name AS creator_name
            FROM images
            LEFT JOIN users ON users.id = images.created_by
            WHERE {alias_where_sql}
            ORDER BY images.id DESC
            LIMIT %s OFFSET %s
            """,
            tuple(alias_params + [page_size, offset]),
        )
        paged["items"] = [self._serialize_image(row) for row in rows]
        return paged

    def overview(self) -> dict:
        total_row = fetch_one(f"SELECT COUNT(*) AS total FROM images WHERE {self._active_gallery_filter()}")
        label_total = len(self.list_label_options())
        batch_row = fetch_one(
            f"""
            SELECT COUNT(*) AS total
            FROM (
                SELECT source, COALESCE(split_name, '') AS split_name
                FROM images
                WHERE {self._active_gallery_filter()}
                GROUP BY source, COALESCE(split_name, '')
            ) batches
            """
        )
        return {
            "imageCount": int((total_row or {}).get("total") or 0),
            "labelCount": label_total,
            "batchCount": int((batch_row or {}).get("total") or 0),
            "storageRoot": serialize_project_path(settings.gallery_root),
        }

    def list_batches(self) -> list[dict]:
        rows = fetch_all(
            f"""
            SELECT source, COALESCE(split_name, '') AS split_name, COUNT(*) AS image_count,
                   MIN(created_at) AS first_created_at, MAX(created_at) AS last_created_at
            FROM images
            WHERE {self._active_gallery_filter()}
            GROUP BY source, COALESCE(split_name, '')
            ORDER BY last_created_at DESC
            """
        )
        batches = [
            {
                "source": row["source"],
                "splitName": row["split_name"] or "",
                "label": self._batch_label(row["source"], row["split_name"] or ""),
                "imageCount": int(row["image_count"]),
                "firstCreatedAt": str(row["first_created_at"]),
                "lastCreatedAt": str(row["last_created_at"]),
            }
            for row in rows
        ]
        batches.sort(key=self._batch_sort_key)
        return batches

    def list_batches_page(self, page: int, page_size: int) -> dict:
        return paginate(self.list_batches(), page, page_size)

    def list_test_group_options(self) -> list[dict]:
        manifest_path = settings.project_root / "data" / "manifests" / "cifar10_test.csv"
        if not manifest_path.exists():
            return []
        frame = read_csv(manifest_path)
        total = len(frame)
        group_count = (total + 99) // 100
        options = []
        for index in range(group_count):
            start = index * 100
            end = min(start + 100, total)
            options.append(
                {
                    "label": f"test{index + 1}",
                    "value": f"test{index + 1}",
                    "startIndex": start,
                    "endIndex": end - 1,
                    "imageCount": end - start,
                }
            )
        return options

    def import_test_group(self, group_name: str, creator_id: int | None, skip_existing: bool = True) -> dict[str, int | str]:
        group_name = str(group_name or "").strip().lower()
        if not group_name.startswith("test"):
            raise ValueError("请选择测试集分组")
        try:
            group_index = int(group_name.replace("test", "", 1))
        except ValueError as exc:
            raise ValueError("测试集分组格式不正确") from exc
        if group_index < 1:
            raise ValueError("测试集分组格式不正确")

        manifest_path = settings.project_root / "data" / "manifests" / "cifar10_test.csv"
        frame = read_csv(manifest_path)
        start = (group_index - 1) * 100
        end = min(start + 100, len(frame))
        if start >= len(frame):
            raise ValueError("测试集分组不存在")

        imported = 0
        skipped = 0
        for _, row in frame.iloc[start:end].iterrows():
            original_name = Path(str(row["file_path"])).name
            if skip_existing and self._exists_in_batch(original_name, "test_set", group_name):
                skipped += 1
                continue
            self.create_from_file(
                image_path=str(row["file_path"]),
                original_name=original_name,
                creator_id=creator_id,
                source="test_set",
                label_name=str(row.get("label_name") or ""),
                split_name=group_name,
            )
            imported += 1
        return {"groupName": group_name, "imported": imported, "skipped": skipped}

    def ensure_label_categories(self) -> None:
        rows = fetch_all(
            f"""
            SELECT DISTINCT label_name
            FROM images
            WHERE {self._active_gallery_filter()} AND label_name IS NOT NULL AND label_name <> ''
            """
        )
        for row in rows:
            label_name = self._normalize_label_name(row.get("label_name"))
            if not label_name:
                continue
            execute("INSERT IGNORE INTO label_categories (name) VALUES (%s)", (label_name,))

    def list_label_options(self) -> list[dict[str, str]]:
        self.ensure_label_categories()
        rows = fetch_all(
            """
            SELECT name
            FROM label_categories
            ORDER BY name ASC
            """
        )
        options = [{"label": self.UNSET_LABEL_NAME, "value": self.UNSET_LABEL_VALUE}]
        options.extend({"label": row["name"], "value": row["name"]} for row in rows)
        return options

    def list_label_categories(self) -> list[dict]:
        self.ensure_label_categories()
        unset_row = fetch_one(
            f"""
            SELECT COUNT(*) AS total
            FROM images
            WHERE {self._active_gallery_filter()} AND (label_name IS NULL OR label_name = '')
            """
        )
        rows = fetch_all(
            f"""
            SELECT label_categories.id, label_categories.name, COUNT(images.id) AS image_count
            FROM label_categories
            LEFT JOIN images
                ON images.label_name = label_categories.name
                AND {self._active_gallery_filter('images')}
            GROUP BY label_categories.id, label_categories.name
            ORDER BY label_categories.name ASC
            """
        )
        categories = [
            {
                "id": self.UNSET_LABEL_VALUE,
                "name": self.UNSET_LABEL_NAME,
                "imageCount": int((unset_row or {}).get("total") or 0),
                "system": True,
            }
        ]
        categories.extend(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "imageCount": int(row["image_count"]),
                "system": False,
            }
            for row in rows
        )
        return categories

    def create_label_category(self, name: str) -> dict:
        normalized = self._normalize_category_name(name)
        execute("INSERT INTO label_categories (name) VALUES (%s)", (normalized,))
        return {"name": normalized}

    def update_label_category(self, category_id: int, name: str) -> dict:
        normalized = self._normalize_category_name(name, ignore_id=category_id)
        row = fetch_one("SELECT id, name FROM label_categories WHERE id = %s LIMIT 1", (category_id,))
        if not row:
            raise ValueError("分类不存在")
        old_name = row["name"]
        execute("UPDATE label_categories SET name = %s WHERE id = %s", (normalized, category_id))
        execute("UPDATE images SET label_name = %s WHERE label_name = %s", (normalized, old_name))
        return {"id": category_id, "name": normalized}

    def delete_label_category(self, category_id: int) -> dict[str, int]:
        row = fetch_one("SELECT id, name FROM label_categories WHERE id = %s LIMIT 1", (category_id,))
        if not row:
            raise ValueError("分类不存在")
        label_name = row["name"]
        execute("UPDATE images SET label_name = NULL WHERE label_name = %s", (label_name,))
        execute("DELETE FROM label_categories WHERE id = %s", (category_id,))
        return {"updated": 1}

    def delete_image(self, image_id: int):
        row = fetch_one("SELECT * FROM images WHERE id = %s LIMIT 1", (image_id,))
        if not row:
            raise ValueError("图片不存在")
        execute("UPDATE images SET is_deleted = 1 WHERE id = %s", (image_id,))
        for field in ("file_path", "thumbnail_path", "feature_path"):
            file_path = resolve_project_path(row[field])
            if file_path.exists():
                file_path.unlink()

    def delete_batch(self, source: str, split_name: str | None) -> dict[str, int]:
        rows = fetch_all(
            """
            SELECT id
            FROM images
            WHERE is_deleted = 0 AND source = %s AND COALESCE(split_name, '') = %s
            ORDER BY id ASC
            """,
            (source, split_name or ""),
        )
        deleted = 0
        for row in rows:
            self.delete_image(int(row["id"]))
            deleted += 1
        return {"deleted": deleted}

    def delete_batches(self, batches: list[dict]) -> dict[str, int]:
        deleted = 0
        for item in batches:
            source = str(item.get("source") or "")
            if not source:
                continue
            result = self.delete_batch(source=source, split_name=str(item.get("splitName") or ""))
            deleted += int(result["deleted"])
        return {"deleted": deleted}

    def rebuild_features(self) -> dict[str, int]:
        rows = fetch_all(
            """
            SELECT id, file_path, feature_path
            FROM images
            WHERE is_deleted = 0
            ORDER BY id ASC
            """
        )
        feature_service = get_feature_service()
        updated = 0
        skipped = 0
        for row in rows:
            image_path = resolve_project_path(row["file_path"])
            feature_path = resolve_project_path(row["feature_path"])
            if not image_path.exists():
                skipped += 1
                continue
            feature = feature_service.extract(str(image_path), mode="baseline")
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, feature.astype(np.float32))
            execute(
                """
                UPDATE images
                SET feature_model = %s, feature_dim = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                ("resnet101_imagenet_pooling", int(feature.shape[0]), int(row["id"])),
            )
            updated += 1
        return {"updated": updated, "skipped": skipped}

    def load_feature_rows(self, include_thumbnails: bool = True) -> list[dict]:
        rows = self._feature_rows()
        payload = []
        for row in rows:
            feature_path = resolve_project_path(row["feature_path"])
            if not feature_path.exists():
                continue
            feature_stat = feature_path.stat()
            payload.append(
                {
                    "id": int(row["id"]),
                    "originalName": row["original_name"],
                    "labelName": row["label_name"],
                    "filePath": serialize_project_path(row["file_path"]),
                    "feature": np.load(feature_path).astype(np.float32),
                    "featurePath": serialize_project_path(feature_path),
                    "featureSize": int(feature_stat.st_size),
                    "featureUpdatedAt": int(feature_stat.st_mtime_ns),
                    "featureModel": row["feature_model"],
                    "featureDim": int(row["feature_dim"] or 0),
                    "thumbnail": thumbnail_to_data_url(row["thumbnail_path"]) if include_thumbnails else None,
                    "thumbnailPath": serialize_project_path(row["thumbnail_path"]),
                    "source": row["source"],
                    "createdAt": str(row["created_at"]),
                }
            )
        return payload

    def load_feature_metadata_rows(self) -> list[dict]:
        payload = []
        for row in self._feature_rows():
            feature_path = resolve_project_path(row["feature_path"])
            if not feature_path.exists():
                continue
            feature_stat = feature_path.stat()
            payload.append(
                {
                    "id": int(row["id"]),
                    "originalName": row["original_name"],
                    "labelName": row["label_name"],
                    "featurePath": serialize_project_path(feature_path),
                    "featureSize": int(feature_stat.st_size),
                    "featureUpdatedAt": int(feature_stat.st_mtime_ns),
                    "thumbnailPath": serialize_project_path(row["thumbnail_path"]),
                    "source": row["source"],
                    "createdAt": str(row["created_at"]),
                }
            )
        return payload

    @staticmethod
    def feature_signature() -> str:
        row = fetch_one(
            f"""
            SELECT
                COUNT(*) AS total_count,
                COALESCE(SUM(id), 0) AS id_sum,
                COALESCE(SUM(feature_dim), 0) AS dim_sum,
                COALESCE(MAX(updated_at), '') AS max_updated_at,
                COALESCE(MAX(id), 0) AS max_id
            FROM images
            WHERE {GalleryService._active_gallery_filter()}
            """
        ) or {}
        payload = "|".join(
            str(row.get(key) or "")
            for key in ("total_count", "id_sum", "dim_sum", "max_updated_at", "max_id")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _feature_rows() -> list[dict]:
        return fetch_all(
            f"""
            SELECT id, original_name, label_name, file_path, feature_path, thumbnail_path, source, created_at,
                   feature_model, feature_dim
            FROM images
            WHERE {GalleryService._active_gallery_filter()}
            ORDER BY id ASC
            """
        )

    @staticmethod
    def _active_gallery_filter(alias: str = "images") -> str:
        prefix = f"{alias}." if alias else ""
        return f"""
            {prefix}is_deleted = 0
            AND (
                {prefix}source <> 'cifar10'
                OR {prefix}id IN (
                    SELECT MAX(dedup.id)
                    FROM images dedup
                    WHERE dedup.is_deleted = 0 AND dedup.source = 'cifar10'
                    GROUP BY dedup.source, dedup.split_name, dedup.original_name, dedup.label_name
                )
            )
        """

    @staticmethod
    def _gallery_where(alias: str = "images", batch_key: str = "", label_name: str = "") -> tuple[str, list[str]]:
        where = GalleryService._active_gallery_filter(alias)
        params: list[str] = []
        prefix = f"{alias}." if alias else ""
        source, split_name = GalleryService._parse_batch_key(batch_key)
        if source:
            where += f" AND {prefix}source = %s AND COALESCE({prefix}split_name, '') = %s"
            params.extend([source, split_name])
        if label_name:
            normalized_label = GalleryService._normalize_label_name(label_name)
            if normalized_label:
                where += f" AND {prefix}label_name = %s"
                params.append(normalized_label)
            else:
                where += f" AND ({prefix}label_name IS NULL OR {prefix}label_name = '')"
        return where, params

    @staticmethod
    def _parse_batch_key(batch_key: str) -> tuple[str, str]:
        value = str(batch_key or "")
        if "||" not in value:
            return "", ""
        source, split_name = value.split("||", 1)
        return source.strip(), split_name.strip()

    @classmethod
    def _normalize_label_name(cls, label_name: str | None) -> str | None:
        value = str(label_name or "").strip()
        if not value or value == cls.UNSET_LABEL_VALUE or value == cls.UNSET_LABEL_NAME:
            return None
        return value

    @classmethod
    def _normalize_category_name(cls, name: str, ignore_id: int | None = None) -> str:
        value = str(name or "").strip()
        if not value:
            raise ValueError("请输入分类名称")
        if value in {cls.UNSET_LABEL_VALUE, cls.UNSET_LABEL_NAME}:
            raise ValueError("未设置分类不能修改")
        existing = fetch_one("SELECT id FROM label_categories WHERE name = %s LIMIT 1", (value,))
        if existing:
            if ignore_id is None or int(existing["id"]) != int(ignore_id):
                raise ValueError("分类已存在")
        return value

    @staticmethod
    def _batch_label(source: str, split_name: str) -> str:
        if source == "test_set" and split_name:
            return split_name
        if source == "upload":
            return split_name or "本地上传"
        if source == "cifar10_test":
            return "固定测试图库"
        if split_name:
            return f"{source}/{split_name}"
        return source

    @staticmethod
    def _batch_sort_key(item: dict) -> tuple[int, int, str]:
        source = str(item.get("source") or "")
        split_name = str(item.get("splitName") or "")
        if source == "test_set" and split_name.startswith("test"):
            try:
                return (0, int(split_name.replace("test", "", 1)), split_name)
            except ValueError:
                return (0, 9999, split_name)
        return (1, 0, str(item.get("lastCreatedAt") or ""))

    @staticmethod
    def _exists_in_batch(original_name: str, source: str, split_name: str) -> bool:
        row = fetch_one(
            """
            SELECT id
            FROM images
            WHERE original_name = %s AND source = %s AND COALESCE(split_name, '') = %s AND is_deleted = 0
            LIMIT 1
            """,
            (original_name, source, split_name or ""),
        )
        return bool(row)

    @staticmethod
    def _serialize_image(row: dict) -> dict:
        return {
            "id": int(row["id"]),
            "originalName": row["original_name"],
            "storedName": row["stored_name"],
            "source": row["source"],
            "labelName": row["label_name"],
            "splitName": row["split_name"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "thumbnail": thumbnail_to_data_url(row["thumbnail_path"]),
            "mimeType": row["mime_type"],
            "featureModel": row["feature_model"],
            "featureDim": int(row["feature_dim"]),
            "createdBy": row.get("creator_name"),
            "createdAt": str(row["created_at"]),
        }
