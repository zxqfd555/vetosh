"""Async Milvus accessor using the ``pymilvus`` async client.

Assumes the indexer wrote chunks into a Milvus collection with fields ``text``
(VARCHAR), ``metadata`` (JSON) and ``embedding`` (FLOAT_VECTOR). Search uses the
``COSINE`` metric so the returned distance *is* the cosine similarity.
"""

from __future__ import annotations

import json
from typing import Any

from serviette.server.accessors.abstract import AsyncVectorAccessor


class MilvusAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._uri = config.resolved_uri()
        self._token = config.token
        self._collection = config.collection
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            from pymilvus import AsyncMilvusClient

            kwargs: dict[str, Any] = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            self._client = AsyncMilvusClient(**kwargs)
            # A collection must be loaded into memory before it can be searched.
            await self._client.load_collection(self._collection)
        return self._client

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        hits = await client.search(
            collection_name=self._collection,
            data=[embedding],
            limit=k,
            output_fields=["text", "metadata"],
            search_params={"metric_type": "COSINE"},
        )
        results: list[dict[str, Any]] = []
        # ``hits`` is a list (one entry per query vector) of lists of hits.
        for hit in hits[0] if hits else []:
            entity = hit.get("entity", hit)
            metadata = entity.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                {
                    "text": entity.get("text", ""),
                    "metadata": metadata or {},
                    "score": float(hit.get("distance", hit.get("score", 0.0))),
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        client = await self._ensure_client()
        info = await client.get_collection_stats(self._collection)
        count = info.get("row_count") if isinstance(info, dict) else None
        return {"chunks": int(count)} if count is not None else {}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
