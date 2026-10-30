"""File I/O with explicit paths, checksums, and JSON-safe numeric output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks.

    Args:
        path: Existing input file.

    Returns:
        Lowercase SHA-256 digest for reproducibility auditing.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """Convert numpy scalars and nonfinite numbers to valid JSON values.

    Args:
        value: Nested metadata or experiment results.

    Returns:
        JSON-compatible structure; nonfinite values become null.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(value: object, path: Path) -> None:
    """Write readable JSON without allowing NaN literals.

    Args:
        value: JSON-serialisable result or configuration.
        path: Destination filename; parent folders are created.

    Returns:
        None; the file is written to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_table(table: pd.DataFrame, path: Path) -> None:
    """Export a portable UTF-8 CSV without a pandas index.

    Args:
        table: Aggregated results, not customer-level raw observations.
        path: Destination CSV filename.

    Returns:
        None; the destination parent is created as necessary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
