"""SQLite vector accessor — TEST ONLY (not part of the public API/docs).

Stores each chunk as a row ``(id, text, metadata JSON, embedding JSON)`` and
performs retrieval with a full-table linear scan, computing cosine similarity in
Python and returning the top-k. This exists purely so the end-to-end
indexer→server flow can be tested without any external service; it is not meant
for real workloads.

The table layout is shared with the indexer's SQLite sink (``graph.py``) via the
constants and helpers below, so both sides agree on the schema.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from vetosh.server.accessors.abstract import AsyncVectorAccessor

DEFAULT_TABLE = "vetosh_embeddings"


def create_table_sql(table: str) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {table} ("
        "  id TEXT PRIMARY KEY,"
        "  text TEXT NOT NULL,"
        "  metadata TEXT NOT NULL,"
        "  embedding TEXT NOT NULL"
        ")"
    )


def connect(path: str | Path, table: str = DEFAULT_TABLE) -> sqlite3.Connection:
    """Open (creating if needed) a SQLite vector store and ensure the schema."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(create_table_sql(table))
    conn.commit()
    return conn


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SqliteAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        # ``config`` is a SqliteConfig; also accept a bare path for convenience.
        if isinstance(config, (str, Path)):
            self.path = str(config)
            self.table = DEFAULT_TABLE
        else:
            self.path = config.path
            self.table = config.table
        self._conn = connect(self.path, self.table)

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        # The scan is synchronous SQLite work; run it off the event loop so the
        # accessor honours its async contract even though SQLite is local.
        return await asyncio.to_thread(self._scan, embedding, k)

    def _scan(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"SELECT text, metadata, embedding FROM {self.table}"
        ).fetchall()
        scored = [
            {
                "text": text,
                "metadata": json.loads(metadata),
                "score": cosine_similarity(embedding, json.loads(emb)),
            }
            for text, metadata, emb in rows
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    async def close(self) -> None:
        self._conn.close()
