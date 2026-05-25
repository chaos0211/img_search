from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "features"
BASELINE_DIM = 2048
EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class MovePlan:
    source: Path
    target: Path
    reason: str


FEATURE_DIRS = {
    ("gallery", "baseline"): FEATURE_ROOT / "cifar10_gallery_baseline",
    ("gallery", "embedding"): FEATURE_ROOT / "cifar10_gallery_embedding",
    ("query", "baseline"): FEATURE_ROOT / "cifar10_query_baseline",
    ("query", "embedding"): FEATURE_ROOT / "cifar10_query_embedding",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="修复 features 下 gallery/query 特征文件错位问题。默认只预览。")
    parser.add_argument("--apply", action="store_true", help="执行移动；不加该参数只输出计划")
    parser.add_argument("--limit", type=int, default=40, help="最多打印多少条移动明细，0 表示全部打印")
    args = parser.parse_args()

    plans = build_plans()
    printable = plans if args.limit == 0 else plans[: args.limit]
    for plan in printable:
        status = "MOVE" if args.apply else "DRY-RUN"
        print(f"[{status}] {rel(plan.source)} -> {rel(plan.target)} | {plan.reason}")
        if args.apply:
            move_path(plan.source, plan.target)
    if args.apply and len(printable) < len(plans):
        for plan in plans[len(printable) :]:
            move_path(plan.source, plan.target)
    if len(printable) < len(plans):
        print(f"... {len(plans) - len(printable)} more")
    print({"plannedMoves": len(plans), "applied": args.apply})


def build_plans() -> list[MovePlan]:
    plans: list[MovePlan] = []
    for (partition, mode), directory in FEATURE_DIRS.items():
        if not directory.exists():
            continue
        for source in sorted(directory.glob("*.npy")):
            expected_partition = expected_partition_for_name(source.name) or partition
            expected_mode = expected_mode_for_file(source) or mode
            if expected_partition == partition and expected_mode == mode:
                continue
            target_dir = FEATURE_DIRS[(expected_partition, expected_mode)]
            target = target_dir / source.name
            if target.exists():
                target = FEATURE_ROOT / "_legacy" / "misplaced_features" / directory.name / source.name
                reason = "目标目录已有同名特征，移入 legacy 避免重复混淆"
            else:
                reason = f"{source.name} 属于 {expected_partition}/{expected_mode}，不属于 {partition}/{mode}"
            plans.append(MovePlan(source=source, target=target, reason=reason))
    return plans


def expected_partition_for_name(filename: str) -> str | None:
    if filename.startswith("train_"):
        return "gallery"
    if filename.startswith("test_"):
        return "query"
    return None


def expected_mode_for_file(path: Path) -> str | None:
    try:
        shape = np.load(path, mmap_mode="r").shape
    except Exception:
        return None
    if len(shape) == 1 and int(shape[0]) == BASELINE_DIM:
        return "baseline"
    if len(shape) == 1 and int(shape[0]) == EMBEDDING_DIM:
        return "embedding"
    return None


def move_path(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def rel(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False)).as_posix()
    except Exception:
        return str(path)


if __name__ == "__main__":
    main()
