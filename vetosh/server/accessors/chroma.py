"""Async ChromaDB accessor using the ``chromadb`` HTTP client.

The indexer's ``pw.io.chroma`` sink writes each chunk with the text as the
document and the source metadata serialized to a JSON string (Chroma metadata
values must be scalars). The collection must be created with the cosine
distance (``hnsw:space = cosine``); Chroma returns a *distance*, converted here
to a similarity (``1 - distance``).
"""

from __future__ import annotations

import json
from typing import Any

from vetosh.server.accessors.abstract import AsyncVectorAccessor


class ChromaAccessor(AsyncVectorAccessor):
    def __init__(self, config) -> None:
        self._config = config
        self._client = None
        self._collection = None

    async def _ensure_collection(self):
        if self._collection is None:
            import chromadb

            cfg = self._config
            self._client = await chromadb.AsyncHttpClient(
                host=cfg.host,
                port=cfg.port,
                ssl=cfg.ssl,
                headers=cfg.headers or {},
                tenant=cfg.tenant,
                database=cfg.database,
            )
            self._collection = await self._client.get_collection(cfg.collection)
        return self._collection

    async def retrieve(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        collection = await self._ensure_collection()
        response = await collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results: list[dict[str, Any]] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            raw = (meta or {}).get("metadata")
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            results.append(
                {
                    "text": text or "",
                    "metadata": metadata,
                    "score": 1.0 - float(distance),
                }
            )
        return results

    async def stats(self) -> dict[str, Any]:
        collection = await self._ensure_collection()
        return {"chunks": int(await collection.count())}

    async def close(self) -> None:
        self._client = None
        self._collection = None
