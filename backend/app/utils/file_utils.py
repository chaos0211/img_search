from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import settings

PROJECT_PATH_ANCHORS = (
    "data",
    "storage",
    "outputs",
    "features",
    "checkpoints",
    "backend",
    "frontend",
    "visual-search-engine",
)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_value: str | Path) -> Path:
    path_text = str(path_value)
    path = Path(path_text)
    if path.is_absolute():
        if path.exists():
            return path
        rebound = _rebase_to_project_root(path_text)
        return rebound or path
    rebound = _rebase_to_project_root(path_text)
    if rebound is not None:
        return rebound
    return settings.project_root / path


def serialize_project_path(path_value: str | Path) -> str:
    path = resolve_project_path(path_value)
    try:
        relative = path.resolve(strict=False).relative_to(settings.project_root.resolve(strict=False))
        return relative.as_posix()
    except Exception:
        return str(path_value).replace("\\", "/")


def write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    ensure_parent(path)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path_value: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    options = {"encoding": "utf-8-sig", **kwargs}
    return pd.read_csv(path, **options)


def _rebase_to_project_root(path_text: str) -> Path | None:
    normalized = path_text.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts:
        return None

    project_name = settings.project_root.name
    if project_name in parts:
        project_index = parts.index(project_name)
        suffix = parts[project_index + 1 :]
        if suffix:
            return settings.project_root.joinpath(*suffix)

    for anchor in PROJECT_PATH_ANCHORS:
        if anchor in parts:
            anchor_index = parts.index(anchor)
            return settings.project_root.joinpath(*parts[anchor_index:])
    return None
