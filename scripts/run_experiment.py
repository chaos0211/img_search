from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.experiments.exp1_baseline import run as run_exp1
from backend.app.experiments.exp2_embedding import run as run_exp2
from backend.app.experiments.exp3_rerank import run as run_exp3
from backend.app.experiments.exp4_index_compare import run as run_exp4
from backend.app.experiments.exp5_duplicate_threshold import run as run_exp5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["exp1", "exp2", "exp3", "exp4", "exp5"], required=True)
    parser.add_argument("--gallery-manifest", default=None)
    parser.add_argument("--query-manifest", default=None)
    parser.add_argument("--baseline-gallery-manifest", default=None)
    parser.add_argument("--baseline-query-manifest", default=None)
    parser.add_argument("--embedding-gallery-manifest", default=None)
    parser.add_argument("--embedding-query-manifest", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.exp == "exp1":
        if not args.gallery_manifest or not args.query_manifest:
            raise ValueError("exp1 需要 --gallery-manifest 和 --query-manifest")
        print(run_exp1(args.gallery_manifest, args.query_manifest, args.top_k))
    elif args.exp == "exp2":
        if not all(
            [
                args.baseline_gallery_manifest,
                args.baseline_query_manifest,
                args.embedding_gallery_manifest,
                args.embedding_query_manifest,
            ]
        ):
            raise ValueError("exp2 需要 baseline 和 embedding 两组清单")
        print(
            run_exp2(
                args.baseline_gallery_manifest,
                args.baseline_query_manifest,
                args.embedding_gallery_manifest,
                args.embedding_query_manifest,
                args.top_k,
            )
        )
    elif args.exp == "exp3":
        if not args.gallery_manifest or not args.query_manifest:
            raise ValueError("exp3 需要 --gallery-manifest 和 --query-manifest")
        print(run_exp3(args.gallery_manifest, args.query_manifest, args.top_k))
    elif args.exp == "exp4":
        if not args.gallery_manifest or not args.query_manifest:
            raise ValueError("exp4 需要 --gallery-manifest 和 --query-manifest")
        print(run_exp4(args.gallery_manifest, args.query_manifest, args.top_k))
    else:
        if not args.gallery_manifest:
            raise ValueError("exp5 需要 --gallery-manifest")
        print(run_exp5(args.gallery_manifest))


if __name__ == "__main__":
    main()
