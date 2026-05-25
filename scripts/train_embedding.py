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
    parser.add_argument("--train-manifest", default="data/manifests/cifar10_train.csv")
    parser.add_argument("--validation-manifest", default="data/manifests/cifar10_test.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--save-best-only", default="true", choices=["true", "false"])
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()

    service = EmbeddingTrainingService()
    result = service.train(
        train_manifest_path=args.train_manifest,
        validation_manifest_path=args.validation_manifest,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        optimizer_name=args.optimizer,
        seed=args.seed,
        save_best_only=args.save_best_only == "true",
        freeze_backbone=not args.unfreeze_backbone,
        device_name=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
