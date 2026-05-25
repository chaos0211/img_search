from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.utils.file_utils import read_csv, resolve_project_path, serialize_project_path, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="根据已存在的 .npy 特征文件重建 feature manifest，不执行模型推理")
    parser.add_argument("--source-manifest", required=True, help="原始图片清单，如 data/manifests/cifar10_gallery.csv")
    parser.add_argument("--feature-dir", required=True, help="已有 .npy 目录，如 features/cifar10_gallery_embedding")
    parser.add_argument("--output", required=True, help="输出 feature manifest，如 features/cifar10_gallery_embedding.csv")
    parser.add_argument("--mode", required=True, choices=["baseline", "embedding"])
    args = parser.parse_args()

    source_manifest = resolve_project_path(args.source_manifest)
    feature_dir = resolve_project_path(args.feature_dir)
    output_path = resolve_project_path(args.output)

    source = read_csv(source_manifest)
    exported = []
    missing = []
    for record in source.to_dict("records"):
        feature_path = feature_dir / f"{record['image_id']}.npy"
        if feature_path.exists():
            exported.append({**record, "feature_path": serialize_project_path(feature_path), "feature_mode": args.mode})
        else:
            missing.append(record["image_id"])

    write_csv(output_path, pd.DataFrame(exported))
    print(
        {
            "sourceManifest": serialize_project_path(source_manifest),
            "featureDir": serialize_project_path(feature_dir),
            "output": serialize_project_path(output_path),
            "mode": args.mode,
            "sourceRows": len(source),
            "syncedRows": len(exported),
            "missingRows": len(missing),
        }
    )


if __name__ == "__main__":
    main()
