from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import get_connection
from backend.app.utils.file_utils import read_csv, serialize_project_path, write_csv


CSV_PATH_COLUMNS = ("file_path", "feature_path", "thumbnail_path")
JSON_PATH_KEYS = ("trainManifest", "validationManifest", "checkpointPath", "metricsPath")


def normalize_csv_files() -> int:
    updated = 0
    for csv_path in sorted((PROJECT_ROOT / "data" / "manifests").glob("*.csv")) + sorted((PROJECT_ROOT / "features").glob("*.csv")):
        frame = read_csv(csv_path)
        changed = False
        for column in CSV_PATH_COLUMNS:
            if column not in frame.columns:
                continue
            normalized = frame[column].map(lambda value: serialize_project_path(value) if isinstance(value, str) and value else value)
            if not normalized.equals(frame[column]):
                frame[column] = normalized
                changed = True
        if changed:
            write_csv(csv_path, frame)
            updated += 1
    return updated


def normalize_json_files() -> int:
    updated = 0
    for json_path in sorted((PROJECT_ROOT / "outputs").rglob("*.json")) + sorted((PROJECT_ROOT / "data").rglob("*.json")):
        try:
            text = json_path.read_text(encoding="utf-8")
        except Exception:
            continue
        payload = None
        try:
            import json

            payload = json.loads(text)
        except Exception:
            continue
        changed = _normalize_json_payload(payload)
        if changed:
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1
    return updated


def _normalize_json_payload(value) -> bool:
    changed = False
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in JSON_PATH_KEYS and isinstance(item, str) and item:
                normalized = serialize_project_path(item)
                if normalized != item:
                    value[key] = normalized
                    changed = True
            else:
                changed = _normalize_json_payload(item) or changed
    elif isinstance(value, list):
        for item in value:
            changed = _normalize_json_payload(item) or changed
    return changed


def normalize_database_rows() -> int:
    updated = 0
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, file_path, thumbnail_path, feature_path FROM images")
                rows = cursor.fetchall()
                for row in rows:
                    file_path = serialize_project_path(row["file_path"])
                    thumbnail_path = serialize_project_path(row["thumbnail_path"])
                    feature_path = serialize_project_path(row["feature_path"])
                    if (
                        file_path == row["file_path"]
                        and thumbnail_path == row["thumbnail_path"]
                        and feature_path == row["feature_path"]
                    ):
                        continue
                    cursor.execute(
                        """
                        UPDATE images
                        SET file_path = %s, thumbnail_path = %s, feature_path = %s
                        WHERE id = %s
                        """,
                        (file_path, thumbnail_path, feature_path, row["id"]),
                    )
                    updated += 1
    except Exception:
        return 0
    return updated


def main() -> None:
    csv_count = normalize_csv_files()
    json_count = normalize_json_files()
    db_count = normalize_database_rows()
    print(
        {
            "csvFilesUpdated": csv_count,
            "jsonFilesUpdated": json_count,
            "databaseRowsUpdated": db_count,
        }
    )


if __name__ == "__main__":
    main()
