from __future__ import annotations

from backend.app.services.gallery_service import GalleryService


class UploadService:
    def __init__(self):
        self.gallery_service = GalleryService()

    def upload_image(
        self,
        data_url: str,
        original_name: str,
        creator_id: int | None,
        source: str = "upload",
        label_name: str | None = None,
        split_name: str | None = None,
    ) -> dict:
        return self.gallery_service.create_from_upload(
            data_url=data_url,
            original_name=original_name,
            creator_id=creator_id,
            source=source,
            label_name=label_name,
            split_name=split_name,
        )
