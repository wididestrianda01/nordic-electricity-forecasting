"""Disk-backed Parquet cache for the data-assembly seam.

`assemble_data` is the single network entry point: every fetcher is joined
there, and the backtest consumes its output in memory. Re-running with the
same (zones, start, end) therefore re-fetches every source for no reason.

This module persists that assembled frame to Parquet keyed by source name,
date range, and identifying parameters, so a later call with the same key
reads from disk instead of the network. It implements the *exact-window raw
layer* pattern: historical slices are immutable, so an exact-range hit is
always valid. `refresh=True` bypasses the cache to re-fetch (after a fetcher
fix or schema change). There is deliberately no incremental fetch — the
benchmark window is fixed, and `end` only moves when a fresh slice is wanted,
which is a new key anyway.

The cache lives at `FORECAST_CACHE_DIR` (default `data/cache`, gitignored).
Writes are atomic (temp file + rename) so an interrupted fetch never leaves a
half-written frame that a later run would mistake for a valid hit.

The cache is an optimization, never load-bearing: a read failure degrades to
a miss (re-fetch) and a write failure is swallowed (the freshly-fetched frame
is still returned). Note the Parquet round-trip drops the frame's ``.freq``
attribute; downstream code (feature building and the backtest) uses actual
timestamps, not ``.freq``, so consumers MUST NOT read ``.freq`` from a cached
frame.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR_ENV = "FORECAST_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path("data/cache")


def cache_dir() -> Path:
    """Return the cache root directory (env-overridable)."""
    return Path(os.environ.get(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR))


def _params_fingerprint(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _path(source: str, start: date, end: date, params: dict[str, Any]) -> Path:
    digest = _params_fingerprint(params)
    name = f"{source}__{start.isoformat()}__{end.isoformat()}__{digest}.parquet"
    return cache_dir() / name


def load(source: str, start: date, end: date, params: dict[str, Any]) -> pd.DataFrame | None:
    """Return the cached frame for the key, or None on a miss/read failure.

    A cache read failure degrades to a miss: the caller re-fetches from the
    network rather than crash on an unreadable, corrupt, or schema-drifted
    file.
    """
    path = _path(source, start, end, params)
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.warning("cache read failed for %s; treating as a miss", source, exc_info=True)
        return None


def store(
    source: str, start: date, end: date, params: dict[str, Any], frame: pd.DataFrame
) -> None:
    """Best-effort persist `frame` under the key (atomic temp + rename).

    Never raises: a cache write failure must not discard a freshly-fetched
    frame. The temp file uses a unique ``mkstemp`` name so concurrent writers
    for the same key never clobber each other, and is removed on failure.
    """
    path = _path(source, start, end, params)
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        frame.to_parquet(tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        logger.warning("cache store failed for %s; keeping freshly-fetched frame", source, exc_info=True)
