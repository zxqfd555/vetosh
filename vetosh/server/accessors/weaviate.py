"""Async Weaviate accessor using the v4 ``weaviate-client``.

The indexer's ``pw.io.weaviate`` sink stores the chunk vector as the object
vector and the remaining columns as properties (``chunk_id``, ``text``, and the
source metadata serialized to a JSON string). Weaviate reports a cosine
*distance*; converted here to a similarity (``1 - distance``).
"""

from __future__ import annotations

import json
from typing import Any

from vetosh.server.accessors.abstract import AsyncVectorAccessor


class WeaviateAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._config = config
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            import weaviate
            from weaviate.connect import ConnectionParams

            cfg = self._config
            auth = (
                weaviate.auth.AuthApiKey(cfg.api_key) if cfg.api_key else None
            )
            self._client = weaviate.WeaviateAsyncClient(
                connection_params=ConnectionParams.from_params(
                    http_host=cfg.http_host,
                    http_port=cfg.http_port,
                    http_secure=cfg.http_secure,
                    grpc_host=cfg.grpc_host or cfg.http_host,
                    grpc_port=cfg.grpc_port,
                    grpc_secure=cfg.grpc_secure,
                ),
                auth_client_secret=auth,
            )
            await self._client.connect()
        return self._client

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        client = await self._ensure_client()
        from weaviate.classes.query import MetadataQuery

        collection = client.collections.get(self._config.collection)
        response = await collection.query.near_vector(
            near_vector=list(map(float, embedding)),
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        results: list[dict[str, Any]] = []
        for obj in response.objects:
            props = obj.properties or {}
            raw = props.get("metadata")
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            distance = obj.metadata.distance if obj.metadata else None
            results.append(
                {
                    "text": props.get("text", ""),
                    "metadata": metadata,
                    "score": 1.0 - float(distance) if distance is not None else 0.0,
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        client = await self._ensure_client()
        collection = client.collections.get(self._config.collection)
        result = await collection.aggregate.over_all(total_count=True)
        count = result.total_count
        return {"chunks": int(count)} if count is not None else {}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
