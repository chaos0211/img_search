from __future__ import annotations

from math import ceil
from typing import Any


def paginate(items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    total = len(items)
    total_pages = max(1, ceil(total / page_size)) if page_size else 1
    page = min(max(page, 1), total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "pageSize": page_size,
        "total": total,
        "totalPages": total_pages,
    }
