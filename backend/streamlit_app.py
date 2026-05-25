from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api_server import API_HOST, API_PORT, ensure_api_server

APP_URL = f"http://{API_HOST}:{API_PORT}"
REDIRECT_URL = f"{APP_URL}?from=streamlit8501"


def main() -> None:
    st.set_page_config(page_title="相似图片检索", page_icon="🖼️", layout="wide")
    ensure_api_server()

    st.html(
        f"""
        <meta http-equiv="refresh" content="0; url={REDIRECT_URL}" />
        <script>
        window.location.replace({json.dumps(REDIRECT_URL)});
        </script>
        """
    )
    st.title("相似图片检索")
    st.caption("8501 只是兼容入口，页面会自动跳转到独立前端登录页。")
    st.link_button("如果没有自动跳转，点击打开登录页", REDIRECT_URL, use_container_width=True)
    st.info(f"标准启动命令是 `uv run python start.py`，登录页实际运行在 {APP_URL}。如果你必须保留 8501 作为主入口，请直接把 uvicorn 起在 8501。")
    st.code(f"uv run uvicorn backend.api_server:app --host {API_HOST} --port 8501", language="bash")


if __name__ == "__main__":
    main()
