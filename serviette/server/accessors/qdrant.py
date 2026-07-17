"""Async Qdrant accessor using ``qdrant-client``.

The indexer's ``pw.io.qdrant`` sink stores each chunk as a point whose payload
carries the remaining columns (``chunk_id``, ``text``, ``metadata``). The
collection uses the cosine metric (auto-created that way by the sink), so the
returned score *is* the cosine similarity.
"""

from __future__ import annotations

from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor


class QdrantAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._url = config.rest_url()
        self._api_key = config.api_key
        self._collection = config.collection
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        client = self._ensure_client()
        response = await client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "metadata": metadata,
                    "score": float(point.score),
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        client = self._ensure_client()
        result = await client.count(collection_name=self._collection, exact=False)
        return {"chunks": int(result.count)}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
