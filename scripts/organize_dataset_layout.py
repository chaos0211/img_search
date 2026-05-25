from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MovePlan:
    source: Path
    target: Path
    reason: str


def main() -> None:
    parser = argparse.ArgumentParser(description="整理非主流程数据目录。默认只预览，不移动文件。")
    parser.add_argument("--apply", action="store_true", help="执行移动；不加该参数只输出计划")
    parser.add_argument(
        "--include-legacy-project-data",
        action="store_true",
        help="同时移动 visual-search-engine/data。该目录属于旧工程，默认只标记不移动。",
    )
    args = parser.parse_args()

    plans = build_plans(include_legacy_project_data=args.include_legacy_project_data)
    if not plans:
        print({"plannedMoves": 0, "applied": False})
        return

    for plan in plans:
        status = "MOVE" if args.apply else "DRY-RUN"
        print(f"[{status}] {rel(plan.source)} -> {rel(plan.target)} | {plan.reason}")
        if args.apply:
            move_path(plan.source, plan.target)

    print({"plannedMoves": len(plans), "applied": args.apply})


def build_plans(include_legacy_project_data: bool) -> list[MovePlan]:
    plans: list[MovePlan] = []
    legacy_root = PROJECT_ROOT / "data" / "_legacy" / "torchvision_root_download"
    root_extract = PROJECT_ROOT / "data" / "cifar-10-batches-py"
    root_archive = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"

    if root_extract.exists():
        plans.append(
            MovePlan(
                source=root_extract,
                target=legacy_root / root_extract.name,
                reason="根目录 torchvision 下载残留；主流程使用 data/raw/cifar10",
            )
        )
    if root_archive.exists():
        plans.append(
            MovePlan(
                source=root_archive,
                target=legacy_root / root_archive.name,
                reason="根目录 CIFAR-10 压缩包残留；主流程使用 data/raw/cifar10/cifar-10-python.tar.gz",
            )
        )

    if include_legacy_project_data:
        source = PROJECT_ROOT / "visual-search-engine" / "data"
        if source.exists():
            plans.append(
                MovePlan(
                    source=source,
                    target=PROJECT_ROOT / "legacy" / "visual-search-engine-data",
                    reason="旧工程数据目录，不属于当前主工程交付口径",
                )
            )
    return [plan for plan in plans if plan.source.exists() and not plan.target.exists()]


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
