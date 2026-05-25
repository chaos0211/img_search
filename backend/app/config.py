from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    backend_root: Path = project_root / "backend"
    frontend_dist: Path = backend_root / "frontend_dist"
    storage_root: Path = project_root / "storage"
    gallery_root: Path = storage_root / "gallery"
    query_root: Path = storage_root / "query"
    feature_root: Path = storage_root / "features"
    index_root: Path = storage_root / "indexes"
    thumbnail_root: Path = storage_root / "thumbnails"
    output_root: Path = project_root / "outputs"
    db_host: str = os.getenv("IMG_SEARCH_DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("IMG_SEARCH_DB_PORT", "33304"))
    db_user: str = os.getenv("IMG_SEARCH_DB_USER", "root")
    db_password: str = os.getenv("IMG_SEARCH_DB_PASSWORD", "123456")
    db_name: str = os.getenv("IMG_SEARCH_DB_NAME", "img_search")
    secret_salt: str = os.getenv("IMG_SEARCH_SECRET_SALT", "img-search-salt")
    default_page_size: int = int(os.getenv("IMG_SEARCH_PAGE_SIZE", "12"))
    duplicate_page_size: int = int(os.getenv("IMG_SEARCH_DUP_PAGE_SIZE", "10"))
    user_page_size: int = int(os.getenv("IMG_SEARCH_USER_PAGE_SIZE", "10"))
    cluster_page_size: int = int(os.getenv("IMG_SEARCH_CLUSTER_PAGE_SIZE", "18"))
    feature_dim: int = int(os.getenv("IMG_SEARCH_FEATURE_DIM", "1024"))
    top_k_default: int = int(os.getenv("IMG_SEARCH_TOP_K_DEFAULT", "10"))
    pq_subvectors: int = int(os.getenv("IMG_SEARCH_PQ_SUBVECTORS", "8"))
    pq_clusters: int = int(os.getenv("IMG_SEARCH_PQ_CLUSTERS", "16"))
    hnsw_m: int = int(os.getenv("IMG_SEARCH_HNSW_M", "16"))
    hnsw_ef_construction: int = int(os.getenv("IMG_SEARCH_HNSW_EF_CONSTRUCTION", "200"))
    hnsw_ef_search: int = int(os.getenv("IMG_SEARCH_HNSW_EF_SEARCH", "64"))
    refresh_interval_ms: int = int(os.getenv("IMG_SEARCH_REFRESH_INTERVAL_MS", "4000"))

    @property
    def allowed_image_suffixes(self) -> set[str]:
        return {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


settings = Settings()

for path in (
    settings.gallery_root,
    settings.query_root,
    settings.feature_root,
    settings.index_root,
    settings.thumbnail_root,
    settings.output_root,
):
    path.mkdir(parents=True, exist_ok=True)
