"""Persistent document-parse cache.

Why this exists
---------------
The indexer reads sources with ``pw.io.fs.read(..., format="only_metadata")``,
so the Pathway graph only ever sees file *paths and metadata*, never file
contents (a 1 GB corpus stays a few KB of paths in the graph). The text is
extracted on demand inside a UDF that reads the file from disk.

When a file changes, Pathway emits two diffs for the same path: a removal of the
old row (``-1``) and an addition of the new row (``+1``). The two rows carry
different metadata (e.g. ``modified_at``/``size``). The removal cannot re-read
the *old* bytes — that version is already gone from disk — yet the downstream
vector-DB sink must retract exactly the chunks that were derived from it.

How Pathway already helps
-------------------------
The parse UDF is registered with ``deterministic=False`` (Pathway's default). On
insertion Pathway memoizes the UDF's output; on the matching retraction it
re-emits that stored value negated **without re-running the UDF**. So within a
single process run, deletions are handled natively and the old file is never
re-read. See ``graph.py``.

What this cache adds
--------------------
1. **Cross-restart warmth** — parsed text survives process restarts, so a
   re-discovered (unchanged) file is not re-parsed. This complements Pathway's
   own persistence and is cheaper to reason about for the parse step.
2. **An explicit, bounded store** — entries are evicted when their document is
   removed (driven from a ``pw.io.subscribe`` callback that receives the native
   ``is_addition`` flag), so the cache cannot grow without bound.

TODO(to_stream alternative): An earlier design read the per-record insert/delete
flag inside the parse UDF via ``table.to_stream(upsert_column_name=...)``. That
*does* surface the flag as a plain column, but ``to_stream`` yields an
append-only table — every change becomes a ``+1`` — which silently breaks
automatic retraction of vectors at the sink. We therefore keep the diff-based
flow and obtain ``is_addition`` at the side-effect boundary via
``pw.io.subscribe`` instead, which is the documented native mechanism.

Key/value
---------
* **Key** — ``json.dumps(metadata, sort_keys=True, ensure_ascii=True)``: stable
  and order-independent across runs.
* **Value** — the extracted document text.
* **Backing store** — a single SQLite file (concurrency-safe, no extra deps),
  located at ``$VETOSH_CACHE_DIR`` or ``/tmp/vetosh_cache`` by default.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from vetosh import APP_NAME

_ENV_CACHE_DIR = f"{APP_NAME.upper()}_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path(f"/tmp/{APP_NAME}_cache")
_DB_FILENAME = "parse_cache.sqlite3"


def cache_key(metadata: dict[str, Any]) -> str:
    """Return a stable cache key for a metadata dict.

    ``sort_keys=True`` makes the key independent of dict insertion order;
    ``ensure_ascii=True`` keeps it byte-stable across locales/platforms.
    """

    return json.dumps(metadata, sort_keys=True, ensure_ascii=True)


def default_cache_dir() -> Path:
    env = os.environ.get(_ENV_CACHE_DIR)
    return Path(env) if env else _DEFAULT_CACHE_DIR


class ParseCache:
    """A persistent, process-safe parse cache backed by SQLite.

    The same cache file can be reused across runs; opening an existing file
    simply reuses the stored entries.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else default_cache_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / _DB_FILENAME
        # check_same_thread=False + a lock: the parse UDF and the subscribe
        # eviction callback may run on different threads.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS parse_cache ("
            "  key TEXT PRIMARY KEY,"
            "  text TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    # -- core operations -------------------------------------------------------

    def get(self, metadata: dict[str, Any]) -> str | None:
        key = cache_key(metadata)
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM parse_cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def store(self, metadata: dict[str, Any], text: str) -> None:
        key = cache_key(metadata)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO parse_cache (key, text) VALUES (?, ?)",
                (key, text),
            )
            self._conn.commit()

    def delete(self, metadata: dict[str, Any]) -> str | None:
        """Delete and return the cached text for ``metadata`` (if present).

        Returning the text lets the caller hand the previously parsed value to
        the sink so it can retract the matching vectors, all without re-reading
        the now-removed file.
        """

        key = cache_key(metadata)
        with self._lock:
            row = self._conn.execute(
                "SELECT text FROM parse_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                self._conn.execute("DELETE FROM parse_cache WHERE key = ?", (key,))
                self._conn.commit()
        return row[0] if row else None

    # -- helpers ---------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM parse_cache").fetchone()[0]

    def __contains__(self, metadata: dict[str, Any]) -> bool:
        return self.get(metadata) is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ParseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
