from __future__ import annotations

import argparse
from pathlib import Path


def iter_appledouble_files(root: Path):
    for path in root.rglob("._*"):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove macOS AppleDouble metadata files (._*) from the project.")
    parser.add_argument(
        "--root",
        default=".",
        help="Project root to scan. Defaults to current directory.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    removed = 0
    for path in iter_appledouble_files(root):
        path.unlink(missing_ok=True)
        removed += 1

    print(f"Removed {removed} AppleDouble files under {root}")


if __name__ == "__main__":
    main()
