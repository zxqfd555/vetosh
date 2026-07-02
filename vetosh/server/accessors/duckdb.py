"""DuckDB vector accessor — embedded, zero-setup, in-database vector search.

Reads the table written by the indexer's ``pw.io.duckdb`` snapshot sink:
``(chunk_id, text, metadata, embedding DOUBLE[])``. Retrieval runs entirely
inside DuckDB with ``list_cosine_similarity`` — a vectorized, columnar scan in
native code (no Python loop over rows). For local corpora this answers in
milliseconds without any external service.

Concurrency note: DuckDB allows one read-write process *or* several read-only
processes per database file — never both. A **streaming** indexer holds the
file read-write for its lifetime, so this accessor cannot query the same file
concurrently; index with ``mode: static`` sources (index once, exit, then
serve), or use a client-server backend (qdrant, pgvector, ...) for concurrent
live indexing and serving. Connections here are short-lived and read-only, so
serving never blocks a subsequent indexer run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from vetosh.server.accessors.abstract import AsyncVectorAccessor

logger = logging.getLogger(__name__)

_LOCK_HINT = (
    "Could not open the DuckDB database %r (is a streaming indexer holding it "
    "read-write?). DuckDB allows one writer OR multiple readers per file. "
    "Run the indexer with `mode: static` sources, or switch to a client-server "
    "backend (qdrant, pgvector, ...) for concurrent indexing and serving."
)


class DuckDbAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._path = config.path
        self._table = config.table

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        # DuckDB's Python API is synchronous; run the query off the event loop.
        return await asyncio.to_thread(self._query, embedding, k)

    # A writer flushing with detach_between_batches holds the lock only
    # briefly; ride out that window before declaring the file unreachable.
    _LOCK_RETRIES = 10
    _LOCK_RETRY_DELAY = 0.2  # seconds

    def _connect_with_retry(self):
        import time

        import duckdb

        last_exc: Exception | None = None
        for _ in range(self._LOCK_RETRIES):
            try:
                return duckdb.connect(self._path, read_only=True)
            except duckdb.Error as exc:
                if "lock" not in str(exc).lower():
                    raise
                last_exc = exc
                time.sleep(self._LOCK_RETRY_DELAY)
        logger.error(_LOCK_HINT, self._path)
        raise RuntimeError(_LOCK_HINT % (self._path,)) from last_exc

    def _query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        conn = self._connect_with_retry()
        try:
            rows = conn.execute(
                f"SELECT text, metadata, "
                f"  list_cosine_similarity(embedding, ?::DOUBLE[]) AS score "
                f'FROM "{self._table}" '
                f"WHERE embedding IS NOT NULL "
                f"ORDER BY score DESC NULLS LAST LIMIT ?",
                [list(map(float, embedding)), k],
            ).fetchall()
        finally:
            conn.close()
        results: list[dict[str, Any]] = []
        for text, metadata, score in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "score": float(score) if score is not None else 0.0,
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._stats)

    def _stats(self) -> dict[str, Any]:
        conn = self._connect_with_retry()
        try:
            chunks, documents, last = conn.execute(
                f"SELECT count(*),"
                f"  count(DISTINCT json_extract_string(metadata, '$.path')),"
                f"  max(TRY_CAST(json_extract(metadata, '$.seen_at') AS BIGINT)) "
                f'FROM "{self._table}"'
            ).fetchone()
        finally:
            conn.close()
        out: dict[str, Any] = {"chunks": int(chunks), "documents": int(documents)}
        if last:
            out["last_indexed_at"] = int(last)
        return out

    async def close(self) -> None:
        return None  # connections are per-query
