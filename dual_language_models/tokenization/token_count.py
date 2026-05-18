from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _count_tokens_in_object(obj: Any) -> int:
    """Recursively count token ids stored in common torch.save payload types."""
    if isinstance(obj, torch.Tensor):
        return int(obj.numel())

    if isinstance(obj, (list, tuple)):
        return sum(_count_tokens_in_object(item) for item in obj)

    if isinstance(obj, dict):
        return sum(_count_tokens_in_object(value) for value in obj.values())

    return 0


def count_tokens_in_file(file_path: str | Path) -> int:
    """Count token ids in a single tokenized shard file (.bin)."""
    path = Path(file_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _count_tokens_in_object(payload)


def list_tokenized_files(path: str | Path, pattern: str = "*.bin", recursive: bool = False) -> list[Path]:
    """Resolve one file or find matching tokenized files in a directory."""
    resolved = Path(path)

    if resolved.is_file():
        return [resolved]

    if not resolved.is_dir():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    if recursive:
        files = sorted(p for p in resolved.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in resolved.glob(pattern) if p.is_file())

    if not files:
        raise FileNotFoundError(
            f"No files matching pattern '{pattern}' found in directory: {resolved}"
        )

    return files


def count_tokens(path: str | Path, pattern: str = "*.bin", recursive: bool = False) -> tuple[int, dict[Path, int]]:
    """Count tokens for one file or many files in a directory."""
    files = list_tokenized_files(path, pattern=pattern, recursive=recursive)

    per_file_counts: dict[Path, int] = {}
    for file_path in files:
        per_file_counts[file_path] = count_tokens_in_file(file_path)

    total = sum(per_file_counts.values())
    return total, per_file_counts
