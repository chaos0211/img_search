from __future__ import annotations

import base64
import io
import mimetypes
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from PIL import Image

from backend.app.config import settings
from backend.app.utils.file_utils import resolve_project_path, serialize_project_path


def _split_data_url(data_url: str) -> tuple[str, bytes]:
    header, encoded = data_url.split(",", 1)
    mime_type = header.split(";")[0].replace("data:", "")
    return mime_type, base64.b64decode(encoded)


def save_base64_image(data_url: str, original_name: str, target_dir: Path) -> dict[str, str | int]:
    mime_type, image_bytes = _split_data_url(data_url)
    suffix = Path(original_name).suffix.lower() or mimetypes.guess_extension(mime_type) or ".png"
    if suffix not in settings.allowed_image_suffixes:
        suffix = ".png"

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    stored_path = target_dir / stored_name
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(stored_path)

    thumbnail_name = f"{stored_path.stem}.jpg"
    thumbnail_path = settings.thumbnail_root / thumbnail_name
    thumbnail = image.copy()
    thumbnail.thumbnail((320, 320))
    thumbnail.save(thumbnail_path, format="JPEG", quality=88)

    return {
        "stored_name": stored_name,
        "file_path": serialize_project_path(stored_path),
        "thumbnail_path": serialize_project_path(thumbnail_path),
        "width": image.width,
        "height": image.height,
        "mime_type": mime_type or "image/jpeg",
    }


def thumbnail_to_data_url(path: str | Path) -> str:
    file_path = resolve_project_path(path)
    with file_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def save_query_image(data_url: str, original_name: str) -> dict[str, str]:
    result = save_base64_image(data_url, original_name, settings.query_root)
    return {
        "file_path": str(result["file_path"]),
        "thumbnail_path": str(result["thumbnail_path"]),
        "mime_type": str(result["mime_type"]),
    }


def fetch_remote_image(image_url: str) -> tuple[bytes, str, str]:
    image_url = image_url.strip()
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("图片链接仅支持http或https")

    request = Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ImgSearch/1.0)",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=20) as response:
        mime_type = response.headers.get_content_type() or "image/jpeg"
        chunks: list[bytes] = []
        total_size = 0
        max_size = 15 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 128)
            if not chunk:
                break
            chunks.append(chunk)
            total_size += len(chunk)
            if total_size > max_size:
                raise ValueError("图片文件过大")

    if not mime_type.startswith("image/"):
        raise ValueError("链接内容不是图片")

    original_name = Path(unquote(parsed.path)).name or "url_image"
    suffix = Path(original_name).suffix.lower() or mimetypes.guess_extension(mime_type) or ".jpg"
    if suffix not in settings.allowed_image_suffixes:
        suffix = ".jpg"
    if not Path(original_name).suffix:
        original_name = f"{original_name}{suffix}"

    return b"".join(chunks), mime_type, original_name


def save_query_image_from_url(image_url: str) -> dict[str, str]:
    image_bytes, mime_type, original_name = fetch_remote_image(image_url)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return save_query_image(f"data:{mime_type};base64,{encoded}", original_name)
