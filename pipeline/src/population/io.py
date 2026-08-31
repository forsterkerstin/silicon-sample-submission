"""src/population/io.py

Filesystem, hashing, and structured-logging helpers shared across the
population-construction pipeline. Nothing here is domain-specific (PUMS/CES);
see pums.py / ces.py / raking.py / sampling.py / roster.py for that.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured (timestamp + level + logger name) logging once
    for the whole pipeline. Safe to call more than once; subsequent calls are
    no-ops if handlers are already attached.
    """
    root = logging.getLogger("population")
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under 'population.<name>'."""
    return logging.getLogger(f"population.{name}")


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in chunks so
    multi-hundred-MB raw inputs (csv_pus.zip, the CES CSV) don't need to be
    held in memory at once.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path | str) -> Path:
    """Create `path` (and parents) if it doesn't exist; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path | str, obj: Any, indent: int = 2) -> None:
    """Write `obj` as JSON, creating the parent directory if needed."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str, sort_keys=False)
        f.write("\n")


def read_json(path: Path | str) -> Any:
    """Read and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_metadata(path: Path | str, role: str, operative: bool) -> dict[str, Any]:
    """Describe one raw input file for source_manifest.json: its relative
    path, filename, SHA-256, size, modification timestamp, and its
    declared role/operative-vs-audit-only status in the pipeline.
    """
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p),
        "filename": p.name,
        "sha256": sha256_file(p),
        "size_bytes": stat.st_size,
        "modified_utc": _mtime_iso(stat.st_mtime),
        "role": role,
        "operative": operative,
    }


def _mtime_iso(mtime: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
