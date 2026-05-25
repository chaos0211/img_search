from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.training_service import EmbeddingTrainingService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["baseline", "embedding"], default="embedding")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-resume", action="store_true", help="重新计算所有特征，不复用已有 .npy")
    parser.add_argument("--flush-every", type=int, default=100, help="每处理多少张图片刷新一次 manifest")
    args = parser.parse_args()

    service = EmbeddingTrainingService()
    result = service.extract_features(
        manifest_path=args.manifest,
        output_manifest_name=args.output,
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        resume=not args.no_resume,
        flush_every=args.flush_every,
    )
    print(result)


if __name__ == "__main__":
    main()
