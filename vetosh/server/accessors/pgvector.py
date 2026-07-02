"""Async pgvector accessor using ``asyncpg``.

Assumes the indexer wrote chunks into a table whose columns include ``text``,
``metadata`` (jsonb) and ``embedding`` (a ``vector(n)`` column from the pgvector
extension). Retrieval uses pgvector's cosine-distance operator ``<=>`` and
converts distance to a cosine *similarity* (``1 - distance``) for the response.
"""

from __future__ import annotations

import json
from typing import Any

from vetosh.server.accessors.abstract import AsyncVectorAccessor


def _to_vector_literal(embedding: list[float]) -> str:
    """Render an embedding as a pgvector text literal: ``[1,2,3]``."""

    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class PgVectorAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._dsn = config.connection_string
        self._table = config.table
        self._pool = None

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(dsn=self._dsn)
        return self._pool

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        vec = _to_vector_literal(embedding)
        # ``embedding <=> $1`` is cosine distance (0 = identical). Order ascending
        # by distance and convert to similarity in the projection.
        query = (
            f"SELECT text, metadata, 1 - (embedding <=> $1::vector) AS score "
            f"FROM {self._table} "
            f"ORDER BY embedding <=> $1::vector ASC "
            f"LIMIT $2"
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, vec, k)
        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                {
                    "text": row["text"],
                    "metadata": metadata or {},
                    "score": float(row["score"]),
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT count(*) AS chunks,"
                f"  count(DISTINCT metadata->>'path') AS documents,"
                f"  max((metadata->>'seen_at')::bigint) AS last_indexed_at "
                f"FROM {self._table}"
            )
        out: dict[str, Any] = {
            "chunks": int(row["chunks"]),
            "documents": int(row["documents"]),
        }
        if row["last_indexed_at"]:
            out["last_indexed_at"] = int(row["last_indexed_at"])
        return out

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
