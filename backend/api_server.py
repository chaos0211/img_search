from __future__ import annotations

import os
import secrets
import socket
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.database import init_database
from backend.app.ui_state import build_state, handle_action, initialize_session, offline_pipeline_service
from backend.app.utils.image_utils import fetch_remote_image

API_HOST = os.getenv("IMG_SEARCH_HOST", "127.0.0.1")
API_PORT = int(os.getenv("IMG_SEARCH_PORT", "8000"))
_SESSION_STORE: dict[str, dict[str, Any]] = {}
_SESSION_LOCK = threading.Lock()
_SERVER_THREAD: threading.Thread | None = None
_FRONTEND_INDEX = settings.frontend_dist / "index.html"


class ActionRequest(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Image Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5175",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _new_session() -> tuple[str, dict[str, Any]]:
    session: dict[str, Any] = {}
    initialize_session(session)
    token = secrets.token_urlsafe(24)
    with _SESSION_LOCK:
        _SESSION_STORE[token] = session
    return token, session


def _parse_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def _get_session(token: str | None, create_if_missing: bool) -> tuple[str | None, dict[str, Any]]:
    if token:
        with _SESSION_LOCK:
            session = _SESSION_STORE.get(token)
        if session is not None:
            initialize_session(session)
            return token, session
    if create_if_missing:
        return _new_session()
    session = {}
    initialize_session(session)
    return None, session


def _apply_action(session: dict[str, Any], action_type: str, payload: dict[str, Any]) -> None:
    if action_type in {"noop", "refresh"}:
        return
    try:
        handle_action(session, {"type": action_type, "payload": payload})
    except Exception as exc:
        session["notice_type"] = "error"
        session["notice"] = _project_relative_message(exc)


def _project_relative_message(exc: Exception) -> str:
    message = str(exc)
    project_root = settings.project_root.resolve().as_posix()
    message = message.replace(project_root + "/", "")
    message = message.replace(project_root, ".")
    return message.replace("\\", "/")


@app.on_event("startup")
def _startup() -> None:
    init_database()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/state")
def get_state(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _parse_token(authorization)
    token, session = _get_session(token, create_if_missing=token is not None)
    return {"token": token, "state": build_state(session)}


@app.get("/api/offline-experiment-runtime")
def get_offline_experiment_runtime() -> dict[str, Any]:
    return offline_pipeline_service.experiment_runtime_status()


@app.post("/api/action")
def post_action(request: ActionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _parse_token(authorization)
    should_create_session = request.type in {"login", "register", "logout", "refresh", "noop"}
    token, session = _get_session(token, create_if_missing=should_create_session)
    _apply_action(session, request.type, request.payload)
    return {"token": token, "state": build_state(session)}


@app.get("/api/image-proxy")
def proxy_image(url: str) -> Response:
    try:
        image_bytes, mime_type, _ = fetch_remote_image(url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=image_bytes,
        media_type=mime_type,
        headers={"Cache-Control": "public, max-age=300"},
    )


def _frontend_file(path: str) -> Path:
    frontend_root = settings.frontend_dist.resolve()
    candidate = (frontend_root / path).resolve()
    if candidate.is_file() and (candidate == frontend_root or frontend_root in candidate.parents):
        return candidate
    return _FRONTEND_INDEX


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend_index() -> FileResponse:
    if not _FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run `npm run build` in `frontend/` first.")
    return FileResponse(_FRONTEND_INDEX)


@app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend(path: str) -> FileResponse:
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    if not _FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run `npm run build` in `frontend/` first.")
    return FileResponse(_frontend_file(path))


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def ensure_api_server() -> None:
    global _SERVER_THREAD

    if _port_open(API_HOST, API_PORT):
        return
    if _SERVER_THREAD and _SERVER_THREAD.is_alive():
        return

    def _run() -> None:
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="warning")

    _SERVER_THREAD = threading.Thread(target=_run, daemon=True, name="img-search-api")
    _SERVER_THREAD.start()


def main() -> None:
    uvicorn.run("backend.api_server:app", host=API_HOST, port=API_PORT, log_level="info")


if __name__ == "__main__":
    main()
