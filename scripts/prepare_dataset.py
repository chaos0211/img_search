from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.dataset_service import DatasetService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=50000)
    parser.add_argument("--test-limit", type=int, default=10000)
    parser.add_argument("--gallery-per-class", type=int, default=5000)
    parser.add_argument("--query-per-class", type=int, default=1000)
    parser.add_argument("--clean-existing", action="store_true")
    args = parser.parse_args()

    service = DatasetService()
    result = service.prepare_cifar10(
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        gallery_per_class=args.gallery_per_class,
        query_per_class=args.query_per_class,
        clean_existing=args.clean_existing,
    )
    print(result)


if __name__ == "__main__":
    main()
