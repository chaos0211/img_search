from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "project_layout_audit.json"
CIFAR10_EXPECTED = {
    "train": 50000,
    "test": 10000,
    "gallery": 50000,
    "query": 10000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="审计主项目数据布局、特征覆盖和实验完成度")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--no-write", action="store_true", help="不写入 outputs/reports/project_layout_audit.json")
    parser.add_argument("--with-size", action="store_true", help="额外统计目录体积；大数据目录会更慢")
    args = parser.parse_args()

    report = build_report(with_size=args.with_size)
    if not args.no_write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human_report(report, None if args.no_write else REPORT_PATH)


def build_report(with_size: bool = False) -> dict[str, Any]:
    manifests = {
        "train": inspect_manifest(PROJECT_ROOT / "data" / "manifests" / "cifar10_train.csv"),
        "test": inspect_manifest(PROJECT_ROOT / "data" / "manifests" / "cifar10_test.csv"),
        "gallery": inspect_manifest(PROJECT_ROOT / "data" / "manifests" / "cifar10_gallery.csv"),
        "query": inspect_manifest(PROJECT_ROOT / "data" / "manifests" / "cifar10_query.csv"),
    }
    features = inspect_feature_manifests()
    metrics = inspect_metrics()
    training = inspect_training()
    legacy = inspect_legacy_locations(with_size=with_size)
    canonical = inspect_canonical_locations(with_size=with_size)
    completion = completion_status(manifests, features, metrics, training)
    return {
        "projectRoot": str(PROJECT_ROOT),
        "canonicalLayout": canonical,
        "legacyOrOutOfScope": legacy,
        "manifests": manifests,
        "featureManifests": features,
        "training": training,
        "metrics": metrics,
        "completion": completion,
    }


def inspect_canonical_locations(with_size: bool = False) -> dict[str, Any]:
    locations = {
        "rawCifar10": PROJECT_ROOT / "data" / "raw" / "cifar10",
        "processedCifar10": PROJECT_ROOT / "data" / "processed" / "cifar10",
        "galleryCifar10": PROJECT_ROOT / "data" / "gallery" / "cifar10",
        "queryCifar10": PROJECT_ROOT / "data" / "query" / "cifar10",
        "manifests": PROJECT_ROOT / "data" / "manifests",
        "features": PROJECT_ROOT / "features",
        "runtimeStorage": PROJECT_ROOT / "storage",
        "metrics": PROJECT_ROOT / "outputs" / "metrics",
        "checkpoints": PROJECT_ROOT / "checkpoints",
    }
    return {name: inspect_path(path, with_size=with_size) for name, path in locations.items()}


def inspect_legacy_locations(with_size: bool = False) -> dict[str, Any]:
    locations = {
        "rootTorchvisionExtract": PROJECT_ROOT / "data" / "cifar-10-batches-py",
        "rootTorchvisionArchive": PROJECT_ROOT / "data" / "cifar-10-python.tar.gz",
        "legacyProject": PROJECT_ROOT / "visual-search-engine",
        "legacyProjectData": PROJECT_ROOT / "visual-search-engine" / "data",
    }
    return {name: inspect_path(path, with_size=with_size) for name, path in locations.items()}


def inspect_path(path: Path, with_size: bool = False) -> dict[str, Any]:
    return {
        "path": relative(path),
        "exists": path.exists(),
        "fileCount": count_files(path) if path.exists() else 0,
        "sizeBytes": path_size(path) if path.exists() and with_size else None,
    }


def inspect_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": relative(path), "exists": False, "rowCount": 0, "labels": {}, "missingFiles": None}

    frame = read_csv(path)
    labels = frame["label_name"].value_counts().sort_index().to_dict() if "label_name" in frame.columns else {}
    missing_files = None
    if "file_path" in frame.columns:
        missing_files = count_missing_paths(frame["file_path"])
    return {
        "path": relative(path),
        "exists": True,
        "rowCount": int(len(frame)),
        "labels": {str(key): int(value) for key, value in labels.items()},
        "missingFiles": missing_files,
    }


def inspect_feature_manifests() -> dict[str, Any]:
    output: dict[str, Any] = {}
    feature_root = PROJECT_ROOT / "features"
    for manifest_path in sorted(feature_root.glob("*.csv")):
        frame = read_csv(manifest_path)
        feature_dir = feature_root / manifest_path.stem
        npy_paths = set(feature_dir.glob("*.npy"))
        referenced_paths: set[Path] = set()
        missing = 0
        dims: list[str] = []
        if "feature_path" in frame.columns:
            for value in frame["feature_path"].dropna().astype(str).tolist():
                candidate = resolve_path(value)
                referenced_paths.add(candidate)
                if not candidate.exists():
                    missing += 1
            for value in frame["feature_path"].dropna().astype(str).head(3).tolist():
                try:
                    dims.append("x".join(str(part) for part in np.load(resolve_path(value), mmap_mode="r").shape))
                except Exception as exc:
                    dims.append(f"error:{type(exc).__name__}")

        labels = frame["label_name"].value_counts().sort_index().to_dict() if "label_name" in frame.columns else {}
        modes = frame["feature_mode"].value_counts().to_dict() if "feature_mode" in frame.columns else {}
        output[manifest_path.name] = {
            "path": relative(manifest_path),
            "rowCount": int(len(frame)),
            "featureDir": relative(feature_dir),
            "featureFileCount": len(npy_paths),
            "missingFeatureFiles": int(missing),
            "orphanFeatureFiles": max(len(npy_paths - referenced_paths), 0),
            "labels": {str(key): int(value) for key, value in labels.items()},
            "modes": {str(key): int(value) for key, value in modes.items()},
            "sampleDims": dims,
        }
    return output


def inspect_metrics() -> dict[str, Any]:
    metric_root = PROJECT_ROOT / "outputs" / "metrics"
    output: dict[str, Any] = {}
    for path in sorted(metric_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if path.name.startswith("exp"):
            output[path.name] = {
                "path": relative(path),
                "indexType": payload.get("indexType"),
                "topK": payload.get("topK"),
                "rerank": payload.get("rerank"),
                "queryCount": payload.get("queryCount"),
                "mapAtK": payload.get("mapAtK"),
                "recallAtK": payload.get("recallAtK"),
                "averageElapsedMs": payload.get("averageElapsedMs"),
                "indexSizeBytes": payload.get("indexSizeBytes"),
            }
        elif path.name == "duplicate_threshold_scan.json":
            thresholds = payload.get("thresholds", [])
            output[path.name] = {
                "path": relative(path),
                "thresholdCount": len(thresholds),
                "hasPositiveSignal": any(float(item.get("precision", 0)) > 0 or float(item.get("recall", 0)) > 0 for item in thresholds),
            }
    return output


def inspect_training() -> dict[str, Any]:
    history_path = PROJECT_ROOT / "outputs" / "metrics" / "training_history.json"
    runtime_path = PROJECT_ROOT / "outputs" / "metrics" / "training_runtime.json"
    history_payload = read_json(history_path, {})
    runtime_payload = read_json(runtime_path, {})
    history = history_payload.get("history", []) if isinstance(history_payload, dict) else []
    checkpoints = {
        name: inspect_path(PROJECT_ROOT / "checkpoints" / name, with_size=True)
        for name in ("embedding_best.pt", "embedding_latest.pt")
    }
    return {
        "historyPath": relative(history_path),
        "runtimePath": relative(runtime_path),
        "runtimeStatus": runtime_payload.get("status"),
        "runtimeMessage": runtime_payload.get("message"),
        "completedEpochs": len(history),
        "plannedEpochs": history_payload.get("epochs") if isinstance(history_payload, dict) else None,
        "bestEpoch": history_payload.get("bestEpoch") if isinstance(history_payload, dict) else None,
        "latestValAccuracy": history[-1].get("valAccuracy") if history else None,
        "checkpoints": checkpoints,
    }


def completion_status(
    manifests: dict[str, Any],
    features: dict[str, Any],
    metrics: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    dataset_complete = all(manifests[name]["rowCount"] == expected for name, expected in CIFAR10_EXPECTED.items())
    feature_targets = {
        "cifar10_gallery_baseline.csv": CIFAR10_EXPECTED["gallery"],
        "cifar10_query_baseline.csv": CIFAR10_EXPECTED["query"],
        "cifar10_gallery_embedding.csv": CIFAR10_EXPECTED["gallery"],
        "cifar10_query_embedding.csv": CIFAR10_EXPECTED["query"],
    }
    feature_complete = all(features.get(name, {}).get("rowCount") == expected for name, expected in feature_targets.items())
    experiments = [name for name in metrics if name.startswith("exp")]
    experiment_complete = len(experiments) >= 8 and all(
        int(metrics[name].get("queryCount") or 0) >= CIFAR10_EXPECTED["query"]
        for name in experiments
        if name != "duplicate_threshold_scan.json"
    )
    training_complete = training.get("runtimeStatus") == "completed" and int(training.get("completedEpochs") or 0) > 1
    duplicate_signal = bool(metrics.get("duplicate_threshold_scan.json", {}).get("hasPositiveSignal"))
    missing = []
    if not dataset_complete:
        missing.append("CIFAR-10 完整数据清单")
    if not feature_complete:
        missing.append("完整 baseline/embedding 特征 manifest")
    if not training_complete:
        missing.append("可说明模型训练完成的训练记录")
    if not experiment_complete:
        missing.append("基于完整查询集的 exp1-exp4 实验结果")
    if not duplicate_signal:
        missing.append("有效的重复阈值扫描结果")
    return {
        "datasetComplete": dataset_complete,
        "featureComplete": feature_complete,
        "trainingComplete": training_complete,
        "experimentComplete": experiment_complete,
        "duplicateThresholdComplete": duplicate_signal,
        "overallComplete": not missing,
        "missing": missing,
        "nextCommands": next_commands(dataset_complete, feature_complete, training_complete, experiment_complete, duplicate_signal),
    }


def next_commands(
    dataset_complete: bool,
    feature_complete: bool,
    training_complete: bool,
    experiment_complete: bool,
    duplicate_signal: bool,
) -> list[str]:
    commands: list[str] = []
    if not dataset_complete:
        commands.append(
            "uv run python scripts/prepare_dataset.py --train-limit 50000 --test-limit 10000 --gallery-per-class 5000 --query-per-class 1000 --clean-existing"
        )
    if not training_complete:
        commands.append(
            "uv run python scripts/train_embedding.py --train-manifest data/manifests/cifar10_train.csv --validation-manifest data/manifests/cifar10_test.csv --epochs 10 --batch-size 16 --device auto"
        )
    if not feature_complete:
        commands.extend(
            [
                "uv run python scripts/extract_features.py --manifest data/manifests/cifar10_gallery.csv --output cifar10_gallery_baseline.csv --mode baseline",
                "uv run python scripts/extract_features.py --manifest data/manifests/cifar10_query.csv --output cifar10_query_baseline.csv --mode baseline",
                "uv run python scripts/extract_features.py --manifest data/manifests/cifar10_gallery.csv --output cifar10_gallery_embedding.csv --mode embedding --checkpoint checkpoints/embedding_latest.pt",
                "uv run python scripts/extract_features.py --manifest data/manifests/cifar10_query.csv --output cifar10_query_embedding.csv --mode embedding --checkpoint checkpoints/embedding_latest.pt",
            ]
        )
    if not experiment_complete:
        commands.extend(
            [
                "uv run python scripts/run_experiment.py --exp exp1 --gallery-manifest features/cifar10_gallery_baseline.csv --query-manifest features/cifar10_query_baseline.csv",
                "uv run python scripts/run_experiment.py --exp exp2 --baseline-gallery-manifest features/cifar10_gallery_baseline.csv --baseline-query-manifest features/cifar10_query_baseline.csv --embedding-gallery-manifest features/cifar10_gallery_embedding.csv --embedding-query-manifest features/cifar10_query_embedding.csv",
                "uv run python scripts/run_experiment.py --exp exp3 --gallery-manifest features/cifar10_gallery_embedding.csv --query-manifest features/cifar10_query_embedding.csv",
                "uv run python scripts/run_experiment.py --exp exp4 --gallery-manifest features/cifar10_gallery_embedding.csv --query-manifest features/cifar10_query_embedding.csv",
            ]
        )
    if not duplicate_signal:
        commands.append("uv run python scripts/run_experiment.py --exp exp5 --gallery-manifest features/cifar10_gallery_embedding.csv")
    return commands


def print_human_report(report: dict[str, Any], report_path: Path | None) -> None:
    completion = report["completion"]
    print("项目布局审计")
    if report_path:
        print(f"- 报告文件: {relative(report_path)}")
    print(f"- 总体完成: {'是' if completion['overallComplete'] else '否'}")
    if completion["missing"]:
        print("- 待补齐:")
        for item in completion["missing"]:
            print(f"  - {item}")
    if completion.get("nextCommands"):
        print("- 下一步命令:")
        for command in completion["nextCommands"][:8]:
            print(f"  {command}")
        if len(completion["nextCommands"]) > 8:
            print(f"  ... {len(completion['nextCommands']) - 8} more")

    print("\n主数据清单:")
    for name, item in report["manifests"].items():
        print(f"- {name}: {item['rowCount']} 行, 缺失文件 {item['missingFiles']}")

    print("\n特征清单:")
    for name, item in report["featureManifests"].items():
        print(
            f"- {name}: manifest {item['rowCount']} 行, .npy {item['featureFileCount']} 个, "
            f"缺失 {item['missingFeatureFiles']}, 游离 {item['orphanFeatureFiles']}, 维度 {item['sampleDims']}"
        )

    print("\n训练:")
    training = report["training"]
    print(
        f"- 状态 {training['runtimeStatus']}, epoch {training['completedEpochs']}/{training['plannedEpochs']}, "
        f"latestValAccuracy={training['latestValAccuracy']}"
    )

    print("\n历史/非主流程位置:")
    for name, item in report["legacyOrOutOfScope"].items():
        if item["exists"]:
            print(f"- {name}: {item['path']} ({item['fileCount']} files)")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def count_missing_paths(values: pd.Series) -> int:
    missing = 0
    for value in values.dropna().astype(str).tolist():
        if not resolve_path(value).exists():
            missing += 1
    return missing


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False)).as_posix()
    except Exception:
        return str(path)


def count_files(path: Path) -> int:
    if path.is_file():
        return 1
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                pass
    return total


if __name__ == "__main__":
    main()
