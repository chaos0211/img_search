from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
import sys

from PIL import Image
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import init_database
from backend.app.config import settings
from backend.app.services.upload_service import UploadService


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()

    init_database()
    dataset = CIFAR10(root=str(settings.project_root / "data" / "raw" / "cifar10"), train=args.split == "train", download=True)
    uploader = UploadService()

    for index in range(min(args.limit, len(dataset))):
        image, label = dataset[index]
        data_url = image_to_data_url(image)
        uploader.upload_image(
            data_url=data_url,
            original_name=f"cifar10_{args.split}_{index}.png",
            creator_id=1,
            source="cifar10",
            label_name=dataset.classes[label],
            split_name=args.split,
        )
        if (index + 1) % 50 == 0:
            print(f"Imported {index + 1}/{min(args.limit, len(dataset))}")


if __name__ == "__main__":
    main()
