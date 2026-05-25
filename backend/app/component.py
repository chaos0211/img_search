from __future__ import annotations

import os
from pathlib import Path

import streamlit.components.v1 as components

from backend.app.config import settings


def get_component():
    dev_url = os.getenv("IMG_SEARCH_FRONTEND_URL", "http://localhost:5173")
    if Path(settings.frontend_dist / "index.html").exists():
        return components.declare_component("img_search_ui", path=str(settings.frontend_dist))
    return components.declare_component("img_search_ui", url=dev_url)
