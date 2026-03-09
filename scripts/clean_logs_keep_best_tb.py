#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_csv_names(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def collect_delete_targets(
    root: Path,
    keep_dirs: set[str],
    keep_files: set[str],
) -> tuple[list[Path], list[Path]]:
    targets: list[Path] = []

    children = list(root.iterdir())
    child_names = {path.name for path in children}
    has_any_file = any(path.is_file() for path in children)
    has_checkpoint_dir = any(
        path.is_dir() and path.name.startswith("checkpoint-") for path in children
    )
    is_single_run_root = bool(child_names & (keep_dirs | keep_files)) or has_any_file or has_checkpoint_dir
    if is_single_run_root:
        run_dirs = [root]
    else:
        run_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]

    for run_dir in run_dirs:
        for child in sorted(run_dir.iterdir()):
            if child.is_dir() and child.name in keep_dirs:
                continue
            if child.is_file() and child.name in keep_files:
                continue
            targets.append(child)

    return targets, run_dirs


def delete_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean training log folders by keeping only best-model and TensorBoard content. "
            "Default behavior is dry-run (preview only)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder that contains run folders, e.g. logs_cp",
    )
    parser.add_argument(
        "--keep-dirs",
        type=str,
        default="tb_logs,best_model",
        help="Comma-separated directory names to keep inside each run folder",
    )
    parser.add_argument(
        "--keep-files",
        type=str,
        default="model.safetensors,best_model_info.json",
        help="Comma-separated file names to keep inside each run folder",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files/directories. If omitted, only preview",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid root directory: {root}")

    keep_dirs = parse_csv_names(args.keep_dirs)
    keep_files = parse_csv_names(args.keep_files)

    targets, run_dirs = collect_delete_targets(root, keep_dirs, keep_files)

    print(f"root={root}")
    print(f"run_dirs={len(run_dirs)}")
    print(f"keep_dirs={sorted(keep_dirs)}")
    print(f"keep_files={sorted(keep_files)}")
    print(f"delete_count={len(targets)}")

    for path in targets:
        print(f"DELETE {path}")

    if args.apply:
        delete_paths(targets)
        print("done=true")
    else:
        print("done=false (dry-run)")


if __name__ == "__main__":
    main()
